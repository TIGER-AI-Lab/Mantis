from PIL import Image
def preprocess_interleaved_images_and_text(text, images=None):
        """
        Args:
            text (`str`, `List[str]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
                text can contain <image> tokens as the placeholder for the image(s) to be inserted.
            images (`PIL.Image.Image`, `List[PIL.Image.Image]`, `List[List[PIL.Image.Image]]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.
                the number of the images should match the number of <image> tokens in the text.
        
        """
        assert text is not None, "text cannot be None."
            
        if images is not None:
            if isinstance(images, Image.Image):
                images = [images]
            if isinstance(images, list) and isinstance(images[0], Image.Image):
                if isinstance(text, str):
                    images = [images]
                elif isinstance(text, list):
                    if len(text) != len(images):
                        raise ValueError("Invalid input text. Number of texts does not match number of images.")
                    images = [[image] for image in images]
            if isinstance(text, str):
                num_images = len(images[0])    
                num_image_tokens = text.count("<image>")
                if num_image_tokens < num_images:
                    # prepend empty image tokens to text
                    if "USER:" in text:
                        text = text.replace("USER:", "USER:" + "<image>" * (num_images - num_image_tokens), 1)
                    elif "Human:" in text:
                        text = text.replace("Human:", "Human:" + "<image>" * (num_images - num_image_tokens), 1)
                    elif "HUMAN:" in text:
                        text = text.replace("HUMAN:", "HUMAN:" + "<image>" * (num_images - num_image_tokens), 1)
                    else:
                        text = "<image>" * (num_images - num_image_tokens) + text
                    # print("Image Tokens <image> are not provided in the text. Automatically prepending them before the text. This might cause model to behave unexpectedly.")
                elif num_image_tokens > num_images:
                    text = text.split("<image>")
                    for i, t in enumerate(text):
                        if i < num_images:
                            text[i] = t + "<image>"
                    text = "".join(text)
                    print(f"Number of <image> tokens: {num_image_tokens} exceeds number of images: {num_images}. Automatically removing extra tokens at the end of the text.")
                    # raise ValueError("Invalid input text. Number of <image> tokens exceeds number of images.")
                texts = [text]
            elif isinstance(text, list):
                if not isinstance(text[0], str):
                    raise ValueError("Invalid input text. Each element of text must be a string.")
                for i, t in enumerate(text):
                    num_image_tokens = t.count("<image>")
                    num_images = len(images[i])
                    if num_image_tokens < num_images:
                        # prepend empty image tokens to text
                        if "USER:" in t:
                            t = t.replace("USER:", "USER:" + "<image>" * (num_images - num_image_tokens), 1)
                        elif "Human:" in t:
                            t = t.replace("Human:", "Human:" + "<image>" * (num_images - num_image_tokens), 1)
                        elif "HUMAN:" in t:
                            t = t.replace("HUMAN:", "HUMAN:" + "<image>" * (num_images - num_image_tokens), 1)
                        else:
                            t = "<image>" * (num_images - num_image_tokens) + t
                        # print("Image Tokens <image> are not provided in the text. Automatically prepending them before the text. This might cause model to behave unexpectedly.")
                    elif num_image_tokens > num_images:
                        t = t.split("<image>")
                        for j, s in enumerate(t):
                            if j < num_images:
                                t[j] = s + "<image>"
                        t = "".join(t)
                        print(f"Number of <image> tokens: {num_image_tokens} exceeds number of images: {num_images}. Automatically removing extra tokens at the end of the text.")
                        # raise ValueError("Invalid input text. Number of <image> tokens exceeds number of images.")
                    text[i] = t
                texts = text
            else:
                raise ValueError("Invalid input text. text must be a string or a list of strings.")
            assert all([t.count("<image>") == len(images_per_text) for t, images_per_text in zip(texts, images)]), "Number of <image> tokens in text does not match number of images."
            # add image denotation in text before each <image> as "(image {i}: <image>)"
            for i, t in enumerate(texts):
                for j in range(len(images[i])):
                    t = t.replace("<image>", f"(image {j+1}: <Image><IMAGE></Image>)", 1)
                t = t.replace("<IMAGE>", "<image>")
                texts[i] = t
            
            # flatten images
            images = [image for images_per_text in images for image in images_per_text]
        else:
            if isinstance(text, str):
                texts = [text]
            elif isinstance(text, list):
                if not isinstance(text[0], str):
                    raise ValueError("Invalid input text. Each element of text must be a string.")
                texts = text
            else:
                raise ValueError("Invalid input text. text must be a string or a list of strings.")
        
        return texts, images