import logging
import os
import sys

# TO USE:
# sys.path.append('G:/My Drive/code/tools') # path to call_gpt.py and log_love.py
# from log_love import setup_logging
# logger = None
# logger = setup_logging(logger)

def get_highest_caller_name() -> str:
    depth = 1
    highest_caller_name = None
    while True:
        try:
            frame = sys._getframe(depth)
        except ValueError:
            # Reached the top of the stack so break the loop
            break
        calling_file_path = frame.f_code.co_filename
        calling_file_name = os.path.basename(calling_file_path)
        if calling_file_name == "pydevd_runpy.py":  # it's running in the debugger
            break
        calling_file_name_without_ext = os.path.splitext(calling_file_name)[0]
        highest_caller_name = calling_file_name_without_ext
        depth += 1
    return highest_caller_name


def setup_logging(logger: logging.Logger = None) -> logging.Logger:
    # Set level to debug
    logging.getLogger().setLevel(logging.DEBUG)
    # To suppress OpenAI module debug, set logging level of swagger_spec_validator logger to WARNING
    logging.getLogger('swagger_spec_validator').setLevel(logging.WARNING)
    # Use a logger with a specific name
    caller_name = get_highest_caller_name()
    logger = logger or logging.getLogger(caller_name)
    logger.setLevel(logging.DEBUG)
    filename = f"{caller_name}.log"

    try:  # Try to remove the log if it already exists
        if os.path.exists(os.path.join("logs", filename)):
            os.remove(os.path.join("logs", filename))
    except Exception as e:
        pass  # If the log file can't be removed, continue to append to it

    if not logger.handlers:
        try:
            # Create a file handler object
            if not os.path.exists("logs"):
                os.makedirs("logs")
            log_path = os.path.join("logs", filename)
            file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')  # 'a' for append mode
            # Set the log level for the file handler
            file_handler.setLevel(logging.DEBUG)
            # Create a formatter for the file handler that includes the function name
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s', "%Y-%m-%d %I:%M:%S %p")
            # Add the formatter to the file handler
            file_handler.setFormatter(file_formatter)
            # Copy anything at level INFO or higher to the console
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            # Create a simplified formatter for the console handler
            console_formatter = logging.Formatter('%(message)s')
            # Add the formatter to the console handler
            console_handler.setFormatter(console_formatter)
            # Add the file handler and console handler to the logger
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        except Exception as e:
            print(f"Error creating file handler and/or console handler: {e}")
    return logger

def main():
    logger = None
    logger = setup_logging(logger)
    logger.info("Testing!")


if __name__ == "__main__":
    main()