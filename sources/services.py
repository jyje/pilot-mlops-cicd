import logging, os, time
import cv2, urllib.request, numpy as np
import torch, torchvision, torchsummary, tritonclient.grpc
import configs as cfn

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(cfn.RESIZE_SIZE),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: x.repeat(cfn.CHANNEL_SIZE, 1, 1)),
        torchvision.transforms.Normalize(cfn.NORMALIZE_MEAN,cfn.NORMALIZE_STD),
    ]
)


def load_data(
    dataset_root_path: str,
    batch_size: int,
    num_workers: int,
    skip_download_dataset: bool,
):
    """
    Load the data from the upstream dataset or local directory
    """

    logging.debug(f"Loading data from {dataset_root_path}")
    dataset = torchvision.datasets.MNIST(
        root = dataset_root_path,
        train = True,
        download = not skip_download_dataset,
        transform = transform,
    )

    logging.debug(f"Creating dataloader with batch size {batch_size} and num workers {num_workers}")
    dataloader = torch.utils.data.DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers,
    )

    logging.debug(f"Load Data... done\n")
    return dataset, dataloader
    

def train_model(
        dataset: torch.utils.data.Dataset,
        dataloader: torch.utils.data.DataLoader,
        learning_rate: float,
        momentum: float,
        t_max: int,
        num_epochs: int,
        batch_size: int,
        train_fc_layer_only: bool,
        dry_run: bool,
):
    """
    Train the model
    """

    def train(
        model : torch.nn.Module,
        epoch: int,
        data_size: int,
        batch_size: int,
        dataloader : torch.utils.data.DataLoader,
        criterion : torch.nn.Module,
        optimizer : torch.optim.Optimizer,
        dry_run: bool,
    ):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        batch_count = 0
        
        logging.debug(f"Training model... (Epoch {epoch+1}/{num_epochs})")

        if dry_run:
            dry_loss = (num_epochs-epoch-1)/num_epochs
            dry_acc = (epoch+1)/num_epochs
            
            time.sleep(0.3)
            batch_count += batch_size
            logging.info(f"Batch {batch_count}/{data_size} (Partial Loss: {dry_loss:.4f}, Partial Accuracy: {100*dry_acc:.1f}%)")
            return dry_loss, dry_acc

        for images, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            predicted = torch.max(outputs.data, 1)[1]
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            batch_count += batch_size
            logging.info(f"Batch {batch_count}/{data_size} (Partial Loss: {loss.item():.4f}, Partial Accuracy: {100*correct/total:.1f}%)")

        total_loss = train_loss / total
        total_acc = correct / total

        return total_loss, total_acc

    # Model configuration
    model = torchvision.models.resnet18(
        weights = torchvision.models.resnet.ResNet18_Weights.IMAGENET1K_V1
    )

    if train_fc_layer_only:
        for _, param in model.named_parameters():
            param.requires_grad = False

    # Fine-tune the fully connected layer
    model.fc = torch.nn.Linear(model.fc.in_features, len(dataset.classes))

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        params = model.parameters(),
        lr = learning_rate,
        momentum = momentum,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)    

    # Train the model
    data_size = len(dataset)
    optimizer.step()
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, epoch, data_size, batch_size, dataloader, criterion, optimizer, dry_run)
        scheduler.step()

        logging.info(f"Epoch {epoch+1}/{num_epochs} (Loss: {train_loss:.4f}, Accuracy: {100*train_acc:.1f}%)")

    logging.info(f"Model Details:\n{model}")
    logging.info(f"Model Summary by `torchsummary`:")
    torchsummary.summary(
        model = model,
        input_size = (cfn.CHANNEL_SIZE, cfn.RESIZE_SIZE[0], cfn.RESIZE_SIZE[1]),
    )

    logging.debug(f"Train Model... done\n")
    return model
    

def save_model(
    class_count: int,
    model: torch.nn.Module,
    model_root_path: str,
    model_name: str,
    model_version: int,
    input_layer_name: str,
    input_layer_data_type: str,
    output_layer_name: str,
    output_layer_data_type: str,
):
    """
    Save the model
    """

    # Build a TorchScript model
    logging.debug("Build a TorchScript model")
    model_scripted = torch.jit.script(model)

    model_scripted_graph = str(model_scripted.graph)
    logging.info(f"Check the graph of the TorchScript-ed model:\n{model_scripted_graph}")
    
    resnet_no_trained_pattern = "return (%3)"
    if resnet_no_trained_pattern in model_scripted_graph:
        logging.warn(f"THE MODEL IS NOT A FINE-TUNED MODEL: the graph contains `{resnet_no_trained_pattern}`\n{model_scripted_graph}")


    # Set model directory
    logging.debug("Set model directory")
    model_profile_dir = f"{model_root_path}/{model_name}"
    model_config_path = f"{model_profile_dir}/config.pbtxt"

    model_version_dir = f"{model_profile_dir}/{model_version}"
    model_path = f"{model_version_dir}/model.pt"

    if not os.path.exists(model_version_dir):
        os.makedirs(model_version_dir, exist_ok=True)


    # Set model config content
    logging.debug("Set model config content")
    model_config_content = f"""
    name: "{model_name}"
    platform: "pytorch_libtorch"
    max_batch_size: 8
    input [
        {{
            name: {input_layer_name}
            data_type: {input_layer_data_type}
            dims: [ {cfn.CHANNEL_SIZE}, {cfn.RESIZE_SIZE[0]}, {cfn.RESIZE_SIZE[1]} ]
        }}
    ]
    output [
        {{
            name: {output_layer_name}
            data_type: {output_layer_data_type}
            dims: [ {class_count} ]
        }}
    ]
    """

    # Do save model
    logging.debug("Do save model")
    model_scripted.save(model_path) # save model
    with open(model_config_path, "w") as f:
        f.write(model_config_content) # save model_config

    logging.debug("Save Model... done\n")
    return model_path, model_config_path


def connect_server(
    server_host: str,
):
    """
    Connect to the model server
    """
    
    client = tritonclient.grpc.InferenceServerClient(
        url = server_host,
        verbose = False,
    )

    assert client.is_server_live(), f"The model server is not live: {server_host}"
    assert client.is_server_ready(), f"The model server is not ready: {server_host}"

    logging.debug(f"Connected to the model server: {server_host}")

    return client


def url_to_tensor(image_url: str) -> torch.Tensor:
    """
    Download the PNG image from the URL and convert it to a model input tensor.
    """
    # 1. Download the image from the URL
    response = urllib.request.urlopen(image_url)
    image_array = np.asarray(bytearray(response.read()), dtype=np.uint8)
    
    # 2. Decode the image and preprocess it
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    # 흑백으로 변환
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 3채널로 확장 (H,W) -> (H,W,3)
    image = np.stack([image]*3, axis=-1)
    image = cv2.resize(image, (cfn.RESIZE_SIZE[0], cfn.RESIZE_SIZE[1]))
    
    # 3. Normalize the image (0-255 -> 0-1)
    image = image.astype(np.float32) / 255.0
    
    # 4. Convert the dimension: (H, W, C) -> (C, H, W)
    image = np.transpose(image, (2, 0, 1))
    
    # 5. Convert the numpy array to a torch tensor
    tensor = torch.from_numpy(image)
        
    return tensor


def request_to_server(
    client: tritonclient.grpc.InferenceServerClient,
    model_name: str,
    model_version: str,
    image: torch.Tensor,
):
    """
    Request to the model server
    """

    image_numpy = image.unsqueeze(0).numpy()

    logging.debug(f"Image shape: {image_numpy.shape}")

    # Set inputs
    inputs = []
    input_tensor = tritonclient.grpc.InferInput("x.1", image_numpy.shape, "FP32")
    input_tensor.set_data_from_numpy(image_numpy)
    inputs.append(input_tensor)

    # Set outputs
    outputs = []
    outputs.append(tritonclient.grpc.InferRequestedOutput("4"))
    
    # tick
    start_time = time.time()

    # Do inference
    response = client.infer(
        model_name = model_name,
        model_version = model_version,
        inputs = inputs,
        outputs = outputs,
    )

    # tock
    end_time = time.time()
    logging.info(f"Inference time: {(end_time - start_time) * 1000:.2f} [ms]")

    return response


def summarize_response(
    response: tritonclient.grpc.InferResult,
):
    """
    Summarize the response
    """
    output_data = response.as_numpy("4")
    print(output_data)

    probabilities = torch.nn.functional.softmax(torch.from_numpy(output_data), dim=1)
    predicted_class = torch.argmax(probabilities).item()

    logging.info(f"Predicted class: {predicted_class}")
    logging.info(f"Probabilities: {probabilities}")

    print(f"Probability Distribution:")
    for idx, prob in enumerate(probabilities[0]):
        print(f" - Number {idx}: {prob.item()*100:.2f}%")
