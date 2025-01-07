# Shakespeare Text Generation with LSTM

This repository contains a text generation model using **LSTM (Long Short-Term Memory)**, built on **TensorFlow/Keras**, that generates text in the style of Shakespeare. The model is trained on a subset of the complete Shakespeare dataset and can generate coherent and creative text sequences based on a given temperature setting.

## Features

- **Text Generation**: Generates text in the style of Shakespeare using a trained LSTM model.
- **Temperature Control**: Allows users to adjust the "creativity" of the generated text via temperature settings.
- **Pre-trained Model**: A pre-trained model (`text_generator_best.h5`) is provided for immediate use.

## Requirements

Before running the project, make sure you have the following dependencies installed:

- Python 3.x
- TensorFlow >= 2.0
- NumPy
- Keras (included with TensorFlow)

You can install the required dependencies by running the following:

pip install tensorflow numpy

# Getting Started
1. Clone the Repository
bash
Copy code
git clone https://github.com/your-username/shakespeare-textgen.git
cd shakespeare-textgen
'''bash
2. Download the Shakespeare Dataset
The script automatically downloads a subset of the Shakespeare dataset for training.

3. Training the Model
If you want to train the model from scratch, run:

bash
Copy code
python train_model.py
This will train the LSTM model on the dataset, and the best model will be saved as text_generator_best.h5.

4. Generating Text
To generate text, simply use the following code in the Python console or script:

python
Copy code
from textgen import generate_text

# Generate text with a temperature of 0.6
generated_text = generate_text(length=300, temperature=0.6)

# Print the generated text
print(generated_text)
You can modify the temperature (ranging from 0.2 to 1.0) to adjust the randomness and creativity of the output.

# Model Architecture
 - LSTM: The model uses an LSTM layer to capture sequential dependencies in the text data.
 - Embedding Layer: An embedding layer is used to convert the characters into dense vectors, improving memory efficiency and  - performance.
 - Softmax Output: A softmax layer at the output produces a probability distribution over possible next characters.

# Example Output
Here’s an example of generated text using a temperature of 0.6:


--- Generated Text (Temperature: 0.6) ---
That in the ghost of thine with so much hope
To bear the weight of man; and thou art taken
By such a weight of earth, that thou shalt die
The noble men with great minds, who speak their speech,
And in the midst of all their strength doth suffer!

# Contributing
Feel free to open an issue or submit a pull request if you want to contribute to this project. Contributions, bug reports, and suggestions are always welcome!

# License
This project is licensed under the MIT License - see the LICENSE file for details.

# Acknowledgments
The Shakespeare dataset used in this project is provided by TensorFlow.


```bash