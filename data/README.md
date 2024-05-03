# Mantis-Instruct Datasets

Mantis-Instruct (many-image question answering) is a dataset to enhance multimodal language models' understanding of QA that associated with multiple images. 

## Dataset Format

### Training data format:
```json
[
    {
        "id": "000000033471",
        "images": ["image1_path", "image2_path", ...],
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nquestion about this image"
            },
            {
                "from": "gpt",
                "value": "answer1"
            },
            {
                "from": "human",
                "value": "<image>question about this image"
            },
            {
                "from": "gpt",
                "value": "answer2"
            },
            {
                "from": "human",
                "value": "question about the first image by denoting in text by 'first image' or ..."
            },
            {
                "from": "gpt",
                "value": "answer3"
            }
        ]
    },
    ...
]
```
Fields requirements:
- `id`: the unique identifier
- `images`: 
    - the image paths used in this data item. Note that this image path should be relative to the json file directory. 
    - For example, if your json file is located at `data/json_file.json`, and one image is located at `data/images/image1.jpg`, then then image path in this fields should be `images/image1.jpg`.
- `converstaions`: 
    - List of converstaions. each messsage has a `from/role` field indicating it's from human or gpt, and a `value/content` filed storing the raw message. 
    - `<image>` is the placeholder of the images. Note that the `<image>` with be replaced by a series of image patches during the inference, so the positions matter. 
    - Make sure you insert the same number of `<image>` token in total in the conversations, and in the reasonable place.
    - If the reasonable positions of the `<image>` token is not easy to infer, you can simply insert them all in the first message at the beginning. Then in your converstaions, do use words like `the first image`, `image 1`, `the right image`, `the final image`, `the image with ...` to indicates which image(s) it is talking about.


### test dataset format
```json
{
    "id": "id",
    "question_type": "multi-choice",
    "question": "<image> What change can be observed in the second pair of images?",
    "images": [
        ""
    ],
    "options": [
        "The person with umbrella is gone",
        "The person with red coat is moved",
        "The car up at the top left is now missing"
    ],
    "answer": "B",
    "data_source": "original dataset",
    "category": "difference description"
}
```
Fields requirements:
- `id`: the unique identifier
- `question_type`: the question type, can be `multi-choice`
- `question`: the question, with `<image>` token indicating the image(s) used in this question.
- `images`: Same as the training data format.
- `options`: the options for the multi-choice question
- `answer`: the answer for the multi-choice question
- `data_source`: the original dataset where this question comes from
- `category`: the category of this question, can be `difference description`, `logical reasoning` or other reasonable categories.

## Upload datasets to hugging face

To upload a dataset to the hugging face, please run the following python script:
```python
repo_id="TIGER-Lab/Mantis-Instruct"
mode="zip" # "parquet" or "zip"
split=train # train, val, test
python upload_dataset_to_hf.py --repo_id $repo_id --image_upload_mode $mode --split $split \
    --dataset_name mind2web \
    --dataset_file mind2web/data/train.json \
    --image_dir mind2web/data/images
```
- `dataset_name` is a custom dataset name, corresponding to the `config` on the hugging face
- `dataset_file` is the json file containing the data of the dataset
- `image_dir` is the directory containing all the images used by this dataset

For examples, please refer to `upload_datasets.sh`.

Before uploading, also remember to set one environment variable to enable the writing to the hugging face datasets.
```bash
export HF_TOKEN="..." # the token Dongfu gives to you!
```