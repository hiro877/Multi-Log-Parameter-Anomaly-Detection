# Multi-Log-Parameter-Anomaly-Detection
This project focuses on detecting anomalies in parameter sequences using a machine learning approach. It includes training and testing configurations for two datasets and also provides a method to visualize the attention heatmap during model inference.

## Setup and Usage

To run this project, you need to configure different settings for each dataset during training and testing phases.

### 1. Training and Testing

Each dataset requires specific configurations for running the training and testing scripts. These configurations are defined in JSON files, and the appropriate configuration file must be specified during execution.

#### **Dataset 1**

To train and test the model on Dataset 1, use the following commands:

\`\`\`bash
python main_param_ad.py --config configurations/exp3.json
python main_param_ad.py --config configurations/exp6.json
python main_param_ad.py --config configurations/exp7.json
\`\`\`

#### **Dataset 2**

For Dataset 2, use these commands:

\`\`\`bash
python main_param_ad.py --config configurations/exp23.json
python main_param_ad.py --config configurations/exp26.json
python main_param_ad.py --config configurations/exp27.json
\`\`\`

**Note:** You need to modify the content of each configuration file based on whether you're in the training or testing phase.

### 2. Visualizing the Attention Heatmap

To generate a heatmap that visualizes the model's attention mechanism, use the following command:

\`\`\`bash
python visualize_attention.py --log_data_path datasets_for_models/test/param4/test.log --output_dir results/ExplainAbility/ --tokenizer_dir param2/vocab_size_20000 --exp_num exp23 --train_data_num 0050
\`\`\`

This command takes the log data as input, processes it, and outputs the heatmap visualizations in the specified directory.

### 3. Compatitive Unsupervised Models
\`\`\`bash
python compatitive_unsupervised_main.py  --model_type autoencoder --train_file train_data.txt --test_file test_data.txt --tokenizer_file path_to_tokenizer_file --batch_size 256
python compatitive_unsupervised_main.py  --model_type vae --train_file train_data.txt --test_file test_data.txt --tokenizer_file path_to_tokenizer_file --batch_size 256
python compatitive_unsupervised_main.py  --model_type oneclasssvm --train_file train_data.txt --test_file test_data.txt --tokenizer_file path_to_tokenizer_file --batch_size 256
python compatitive_unsupervised_main.py  --model_type isolationforest --train_file train_data.txt --test_file test_data.txt --tokenizer_file path_to_tokenizer_file --batch_size 256
\`\`\`

### Requirements

- Python 3.x
- Required Python libraries (to be added based on your environment)
  
Make sure to install any dependencies mentioned in the \`requirements.txt\` or set up a virtual environment as necessary.

