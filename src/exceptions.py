import sys
import logging
from types import ModuleType

def get_error_message(error, error_detail: ModuleType) -> str:
        """
        Extracts detailed error information including file name, line number, and the error message.
        error: The exception that occurred.
        error_detail: The sys module to access traceback details.
        return: A formatted error message string.
        """
        ## open the error black box - the Traceback
        _, _, exc_tb = error_detail.exc_info()

        if exc_tb is None:
            return str(error)

        ## The Fat Cat Fell
        ## file name and line number where it fail
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno

        ## build the final error message
        error_message = f"Error in the file [{file_name}] at line number [{line_number}]: {str(error)}"
        logging.error(error_message) ##logging error message

        return error_message


## Custom class 
class CustomException(Exception):
      "Custom exception class for handling errors in the IVIM-PICVAE project"
      def __init__(self, error_message: str, error_details: ModuleType):
            # Always call the parent __init__ first with error messages
            super().__init__(error_message)

            # Run our detective function to get the detailed message
            # Format the detailed error message using the error_message_detail function
            self.error_message = get_error_message(error=error_message, error_detail=error_details)

        # The Magic String Method
        # This tells Python what to print when you 'print()' or 'raise' the error
      def __str__(self) -> str:
            return self.error_message
      
