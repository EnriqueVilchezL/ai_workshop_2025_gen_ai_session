import re
import traceback

def handle_exception(e: Exception, model:str)-> None:
    print(f"Error occurred while running the RAG pipeline: {e}")
    print("\nDetailed error information:")
    traceback.print_exc()
    
    # Error message patterns and corresponding help messages
    error_patterns = [
        (r"connect(?:ion)?\s+refused", "Make sure Ollama is running. Try 'ollama serve' in another terminal."),
        (r"no\s+such\s+file\s+or\s+directory", "Check that the required directories exist and contain documents."),
        (r"model\s+not\s+found", f"The model '{model}' might not be downloaded. Run 'ollama pull {model}'."),
        (r"time(?:out|d?\s+out)", "The request timed out. Check your network connection or server load."),
        (r"permission\s+denied", "Check file permissions for the resources being accessed."),
        (r"memory|out\s+of\s+memory", "The operation ran out of memory. Try a smaller model or reduce batch size."),
        (r"api\s+key", "Check that your API key is properly configured and valid."),
    ]
    
    # Check the error message against each pattern
    error_msg = str(e).lower()
    for pattern, help_message in error_patterns:
        if re.search(pattern, error_msg):
            print(f"\nTip: {help_message}")
            break
    else:
        # If no specific pattern matched
        print("\nTip: Check the error message above for details on how to resolve this issue.")