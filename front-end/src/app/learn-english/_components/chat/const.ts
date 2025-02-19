const tools = [{
    "type": "function",
    "function": {
        "name": "correct_message",
        "description": "Correct the user's message.",
        "parameters": {
            "type": "object",
            "properties": {
                "correction": {
                    "type": "string",
                    "description": "Correct the user's message with brief explanation and correct message. If the message is correct, return nothing."
                },
                "correct": {
                    "type": "boolean",
                    "description": "If the message is correct, return true. Otherwise, return false."
                }
            },
            "required": [
                "correction", "correct"
            ],
            "additionalProperties": false
        },
        "strict": true
    }
}];
export default tools;