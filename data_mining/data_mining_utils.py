

def process_image(ls):
    file_dir, url = ls
    # try:
    response = requests.get(url, headers=headers, timeout=(5, 5))
    img = Image.open(BytesIO(response.content))

    old_width, old_height = img.size

    ratio = max(old_width, old_height) / config['max_dim_size']
    new_width, new_height = int(old_width / ratio), int(old_height / ratio)

    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    img = img.resize((new_width, new_height))
    img.save(file_dir, "jpeg", quality=95)
    # except Exception as e:
        # logger.critical(f"Error when processing {file_dir}, url {url}: {traceback.format_exc()}; Request Content is {response.content}")

def trimArgs(path_url):