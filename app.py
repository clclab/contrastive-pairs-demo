import torch
import datasets
import gradio

from transformers import GPT2LMHeadModel, GPT2TokenizerFast


class CrowSPairsDataset(object):
    def __init__(self):
        super().__init__()

        self.df = (datasets
                .load_dataset("BigScienceBiasEval/crows_pairs_multilingual")["test"]
                .to_pandas()
                .query('stereo_antistereo == "stereo"')
                .drop(columns="stereo_antistereo")
            )

    def sample(self, bias_type, n=10):
        return self.df[self.df["bias_type"] == bias_type].sample(n=n)

    def bias_types(self):
        return self.df.bias_type.unique().tolist()


def run(bias_type):
    sample = dataset.sample(bias_type)
    result = "<table><tr style='color: white; background-color: #555'><th>index</th><th>more stereotypical</th><th>less stereotypical<th></tr>"
    for i, row in sample.iterrows():
        result += f"<tr><td>{i}</td>"
        more = row["sent_more"]

        more = tokenizer(more, return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            out_more = model(more, labels=more.clone())
            score_more = out_more["loss"]
            perplexity_more = torch.exp(score_more).item()

        less = row["sent_less"]
        less = tokenizer(less, return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            out_less = model(less, labels=less.clone())
            score_less = out_less["loss"]
            perplexity_less = torch.exp(score_less).item()
            if perplexity_more > perplexity_less:
                shade = round(
                    abs((perplexity_more - perplexity_less) / perplexity_more), 2
                )
                result += f"<td style='padding: 0 1em;)'>{row['sent_more']}</td><td style='padding: 0 1em; background-color: rgba(255,0,255,{shade})'>{row['sent_less']}</td></tr>"
            else:
                shade = abs((perplexity_less - perplexity_more) / perplexity_less)
                result += f"<td style='padding: 0 1em; background-color: rgba(0,255,255,{shade})'>{row['sent_more']}</td><td style='padding: 0 1em;'>{row['sent_less']}</td></tr>"
    result += "</table>"
    return result


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model_id = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
dataset = CrowSPairsDataset()

bias_type_sel = gradio.Dropdown(label="Bias Type", choices=dataset.bias_types())

iface = gradio.Interface(
    fn=run,
    inputs=bias_type_sel,
    outputs="html",
    title="Detecting stereotypes in the GPT-2 language model using CrowS-Pairs",
    description="GPT-2 is a language model which can score how likely it is that some text is a valid English sentence: not only grammaticality, but also the 'meaning' of the sentence is part of this score. CrowS-Pairs is a dataset with pairs of more and less stereotypical examples for different social groups (e.g., gender and nationality stereotypes). We sample 10 random pairs from CrowS-Pairs and show whether the stereotypical example gets a higher score ('is more likely'). If GPT-2 systematically prefers the stereotypical examples, it has probably learnt these stereotypes from the training data.",
)

iface.launch()
