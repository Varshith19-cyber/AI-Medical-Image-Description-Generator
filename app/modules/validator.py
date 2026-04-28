def validate_result(result):
    result["warning"] = "This is an academic assistive output, not a medical diagnosis."
    return result