import argparse, logging, time
from services import connect_server, url_to_tensor, request_to_server, summarize_response

# Logging presets
logging.Formatter.converter = time.gmtime
logging.addLevelName(logging.DEBUG, "\033[94mDEBUG\033[0m")
logging.addLevelName(logging.INFO, "\033[92mINFO\033[0m")
logging.addLevelName(logging.WARNING, "\033[93mWARNING\033[0m")
logging.addLevelName(logging.ERROR, "\033[91mERROR\033[0m")
logging.addLevelName(logging.CRITICAL, "\033[95mCRITICAL\033[0m")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # server-host
    parser.add_argument(
        "--server-host",
        type = str,
        required = True,
        help = "The host of the model server. e.g. localhost:8000"
    )

    # model-name
    parser.add_argument(
        "--model-name",
        type = str,
        required = True,
        help = "The name of the model. e.g. pilot"
    )

    # model-version
    parser.add_argument(
        "--model-version",
        type = str,
        required = True,
        help = "The version of the model. e.g. 1"
    )

    # image-url
    parser.add_argument(
        "--image-url",
        type = str,
        required = True,
        help = "The URL of the image to request to the model server. e.g. https://example.com/image.png"
    )

    # log-level
    parser.add_argument(
        "--log-level",
        type = str,
        required = False,
        default = "INFO",
        choices = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help = "The log level of the model client"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level = args.log_level.upper(),
        format = "%(levelname)s [%(asctime)sZ] %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
    )

    # Display Arguments
    logging.debug(f"Arguments: {dict(args._get_kwargs())}")

    ## Pipeline Start
    logging.debug("model-client.py")
    logging.debug("Pipeline Start")

    # 1. Connect to the model server
    logging.debug(" === Pipeline 1/2. Connect to the model server === ")
    client = connect_server(
        server_host = args.server_host,
    )

    # 2. Normalize the image
    logging.debug(" === Pipeline 2/2. Normalize the image === ")
    image_tensor = url_to_tensor(
        image_url = args.image_url,
    )

    # 3. Request to the model server
    logging.debug(" === Pipeline 3/3. Request to the model server === ")
    response = request_to_server(
        client = client,
        model_name = args.model_name,
        model_version = args.model_version,
        image = image_tensor,
    )

    # 4. Summarize the response
    logging.debug(" === Pipeline 4/4. Summarize the response === ")
    summarize_response(
        response = response,
    )

    # Pipeline End
    logging.debug("Pipeline End")
