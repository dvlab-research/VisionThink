from PIL import Image

def resize_image(image: Image.Image, save_path=None):
    original_width, original_height = image.size
    # Calculate the new dimensions (double the size)
    new_width = original_width * 2
    new_height = original_height * 2
    print(f"[TOOL CALL RESIZE IMAGE]: NEW_IMAGE_SIZE: {(new_width, new_height)}.")
    # Resize the image
    resized_image = image.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)
    if save_path:
        # Save the enlarged image
        resized_image.save(save_path)
    return resized_image

def prepare_tool_call_inputs(json_objects: list):
    for obj in json_objects:
        action_type = obj['arguments']['action']
        assert action_type in ["resize"], f"Unknown Tool Type: {action_type}. Available function tools are: `resize`"
        assert len(json_objects) == 1, f"You should only call function `resize` once per function call."
    return action_type