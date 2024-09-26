import sys

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shap
import argparse
import json
from transformers import BertForMaskedLM, AdamW, BertConfig
from tokenizers import BertWordPieceTokenizer
from captum.attr import IntegratedGradients
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_model_and_tokenizer(tokenizer_dir, exp_num, train_data_num, use_proposed_method):
    # Set up tokenizer and model configuration
    tokenizer_file = f'models/anomaly_detection/parameter/trained_tokenizer/{tokenizer_dir}/vocab.txt'
    tokenizer = BertWordPieceTokenizer(tokenizer_file)

    if use_proposed_method:
        vocab_size_ = 200000
    else:
        vocab_size_=tokenizer.get_vocab_size()
    config = BertConfig(vocab_size=vocab_size_, hidden_size=256, num_hidden_layers=4, num_attention_heads=4,
                        intermediate_size=512)

    # Determine the device to use (CPU or GPU)

    model_path = "saved_models/{}/{}.pth".format(exp_num, train_data_num)
    model = BertForMaskedLM(config).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return tokenizer, model


def visualize_attention(tokenizer, input_ids, attention, output_path):
    attention = attention[0]  # バッチの最初の要素を取得
    attention = torch.mean(attention, dim=1)  # ヘッドごとの平均を取る

    # GPU テンソルを CPU に移動し、NumPy 配列に変換
    attention_np = attention.detach().cpu().numpy()

    # convert_ids_to_tokens の代替
    tokens = [tokenizer.id_to_token(id.item()) for id in input_ids[0]]

    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_np, xticklabels=tokens, yticklabels=tokens, cmap='viridis')
    plt.title("Attention Visualization")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def evaluate_feature_importance(tokenizer, model, input_ids, input_mask, output_path):
    # リストをNumPy配列に変換
    # print(type(input_ids))
    # print(type(input_mask))
    # print(input_ids.shape)
    # sys.exit()
    # input_ids_np = input_ids.cpu().numpy()
    # input_mask_np = np.array(input_mask)

    # 入力をテンソルに変換し、正しい型とデバイスに設定
    input_ids = input_ids.long().to(model.device)
    input_mask = torch.tensor(input_mask, dtype=torch.long).to(model.device)

    # モデルをラップする関数を定義
    def model_forward(input_ids):
        return model(input_ids, attention_mask=input_mask).logits

    # IntegratedGradientsのインスタンスを作成
    ig = IntegratedGradients(model_forward)

    # 属性を計算
    attributions, delta = ig.attribute(input_ids,
                                       target=0,  # 最初のトークンの予測に対する属性を計算
                                       n_steps=50,
                                       return_convergence_delta=True)

    # 属性をCPUに移動し、NumPy配列に変換
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions.cpu().detach().numpy()

    # トークンのデコード
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().cpu().tolist())

    # 結果の可視化
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(tokens)), attributions)
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.title("Feature Importance (Integrated Gradients)")
    plt.xlabel("Tokens")
    plt.ylabel("Attribution")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


    return attributions

def visualize_attention(tokenizer, input_ids, attention, output_path):
    attention = attention[0]  # バッチの最初の要素を取得
    attention = torch.mean(attention, dim=1)  # ヘッドごとの平均を取る

    # GPU テンソルを CPU に移動し、NumPy 配列に変換
    attention_np = attention.detach().cpu().numpy()

    # convert_ids_to_tokens の代替
    tokens = [tokenizer.id_to_token(id.item()) for id in input_ids[0]]

    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_np, xticklabels=tokens, yticklabels=tokens, cmap='viridis')
    plt.title("Attention Visualization")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main(args):
    tokenizer, model = load_model_and_tokenizer(args.tokenizer_dir, args.exp_num, args.train_data_num,
                                                args.use_proposed_method)

    # 分析対象のテキスト
    texts = [
        "state=B, value=[MASK], InfoD",
        "state=B, value=1743, InfoD",
        'state=B, value=46, InfoD',

        "state=C, value=[MASK], InfoN",
        "state=C, value=2327, InfoN",
        "state=C, value=1813, InfoN",
        "state=C, value=1667, InfoN",
        "state=C, value=7875, InfoN",
    ]

    # 対象文字列indexを指定
    texts_ind = 3
    output_list = []
    with torch.no_grad():
        # テキストをトークナイズして、トークンIDを取得
        inputs = tokenizer.encode(texts[texts_ind])
        print(f"Encoded input: {inputs}")

        # 'ids'属性からトークンIDを取得
        input_ids_list = inputs.ids
        print(f"Token IDs: {input_ids_list}")

        # 'tokens'属性からトークンを取得
        tokens = inputs.tokens
        print(f"Tokens: {tokens}")

        # トークンIDをPyTorchテンソルに変換
        input_ids = torch.tensor([input_ids_list]).to(device)

        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)
            predictions = outputs.logits

        # Attentionの取得
        attention = outputs.attentions[-1][0]
        print(f"Attention shape: {attention.shape}")

        # ヘッダーレイヤーを全て足し上げて、40x40のスライスを取得
        sum_attention = torch.sum(attention, dim=0)[:40, :40]
        sum_attention = sum_attention.detach().cpu()

        # 出力をリストに追加
        output_list.append(sum_attention[0].numpy())
        output_list.append(sum_attention[3].numpy())
        output_list.append(sum_attention[7].numpy())
        output_list.append(sum_attention[9].numpy())

        # 列毎の平均を出力リストに追加
        column_sums = sum_attention.mean(dim=0)
        output_list.append(column_sums.numpy())

        # CSVに出力
        df = pd.DataFrame(output_list)
        df.to_csv('output.csv', index=False)

        # ヒートマップの描画（軸ラベルをトークンに設定）
        plt.figure(figsize=(10, 8))
        # sns.heatmap(sum_attention, cmap='viridis', xticklabels=tokens[:40], yticklabels=tokens[:40])

        # カラーバーの目盛りを自動で10個の位置に設定
        cbar_ticks = np.linspace(sum_attention.min(), sum_attention.max(), 10)

        # ヒートマップの描画
        ax = sns.heatmap(sum_attention, cmap='viridis', xticklabels=tokens[:40], yticklabels=tokens[:40],
                         cbar_kws={'label': '', 'ticks': cbar_ticks})

        # カラーバーを取得して、フォントサイズを設定
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=14)  # カラーバーのフォントサイズを14に設定

        # X軸とY軸のラベルフォントサイズを設定
        plt.xticks(rotation=90, fontsize=14)  # X軸ラベルを90度回転し、フォントサイズ14に設定
        plt.yticks(fontsize=14)  # Y軸ラベルのフォントサイズを14に設定

        plt.show()

    # # ログデータの読み込み
    # with open(args.log_data_path, 'r') as f:
    #     log_data = f.readlines()
    #
    # for i, log in enumerate(log_data):
    #     # inputs = tokenizer(log + " " + " ".join(custom_tokens), return_tensors="pt", padding=True, truncation=True)
    #     inputs = tokenizer.encode(log)
    #     encoded_input = inputs.ids
    #     input_ids = torch.tensor([encoded_input]).to(device)
    #     with torch.no_grad():
    #         outputs = model(input_ids, output_attentions=True)
    #         predictions = outputs.logits
    #
    #     # outputs = model(**inputs, output_attentions=True)
    #     # print(tokenizer.encode(log).attention_mask)
    #     visualize_attention(tokenizer, input_ids, outputs.attentions[-1],
    #                         f"{args.output_dir}/attention_viz_{i}.png")
    #
    #     # shap_values = evaluate_feature_importance(tokenizer, model, input_ids, inputs.attention_mask,
    #     #                             f"{args.output_dir}/feature_importance_{i}.png")
    #     # sys.exit()

    print("処理が完了しました。")




    # with open(miss_results_file_path, 'a+') as file:
    #     for text_batch, masked_param_batch, label_batch in self.data_loader:
    #         for i in range(len(text_batch)):
    #             text, masked_param, label = text_batch[i], masked_param_batch[i], label_batch[i]
    #             encoded_input = self.tokenizer.encode(text).ids
    #             mask_index = encoded_input.index(self.tokenizer.token_to_id("[MASK]"))
    #             # encoded_input = self.add_positional_info(encoded_input, mask_index)
    #             # print(encoded_input)
    #             # sys.exit()
    #             input_ids = torch.tensor([encoded_input]).to(self.device)
    #             with torch.no_grad():
    #                 outputs = self.model(input_ids)
    #                 predictions = outputs.logits

"""
python visualize_attention.py --log_data_path datasets/test.log --output_dir results/ExplainAbility/ --tokenizer_dir vocab_size_2558 --exp_num exp4 --train_data_num 0050 --use_proposed_method

python visualize_attention.py --log_data_path datasets/test.log --output_dir results/ExplainAbility/ --tokenizer_dir vocab_size_10051 --exp_num exp7 --train_data_num 0050 --use_proposed_method

python visualize_attention.py --log_data_path datasets_for_models/test/param4/test.log --output_dir results/ExplainAbility/ --tokenizer_dir param2/vocab_size_20000 --exp_num exp27w --train_data_num 0050 --use_proposed_method

"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BERTモデルの注意機構の可視化と特徴量重要度の評価")
    # parser.add_argument("--model_path", type=str, required=True, help="事前学習済みモデルのパス")
    # parser.add_argument("--tokenizer_path", type=str, required=True, help="事前学習済みトークナイザーのパス")
    # parser.add_argument("--custom_tokens_path", type=str, required=True, help="独自トークンのJSONファイルパス")
    parser.add_argument("--log_data_path", type=str, required=True, help="ログデータのファイルパス")
    parser.add_argument("--output_dir", type=str, default="output", help="出力ディレクトリ")

    parser.add_argument("--tokenizer_dir", type=str, help="事前学習済みモデルのパス")
    parser.add_argument("--train_data_num", type=str, help="事前学習済みモデルのパス")
    parser.add_argument("--exp_num", type=str, help="事前学習済みモデルのパス")
    parser.add_argument("--use_proposed_method", action='store_true')

    args = parser.parse_args()
    main(args)
