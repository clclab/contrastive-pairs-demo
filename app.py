import torch
import datasets
import gradio
import pandas

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


def run(df):
    result = "<table><tr style='color: white; background-color: #555'><th>index</th><th>more stereotypical</th><th>less stereotypical<th></tr>"
    for i, row in df.iterrows():
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
                shade = (shade + 0.2) / 1.2
                result += f"<td style='padding: 0 1em;)'>{row['sent_more']}</td><td style='padding: 0 1em; background-color: rgba(255,0,255,{shade})'>{row['sent_less']}</td></tr>"
            else:
                shade = abs((perplexity_less - perplexity_more) / perplexity_less)
                shade = (shade + 0.2) / 1.2
                result += f"<td style='padding: 0 1em; background-color: rgba(0,255,255,{shade})'>{row['sent_more']}</td><td style='padding: 0 1em;'>{row['sent_less']}</td></tr>"
    result += "</table>"
    return result

def sample_and_run(bias_type):
    sample = dataset.sample(bias_type)
    return run(sample)

def manual_run(more, less):
    df = pandas.DataFrame.from_dict({
            'sent_more': [more],
            'sent_less': [less],
            'bias_type': ["manual"],
        })
    return run(df)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model_id = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
dataset = CrowSPairsDataset()

bias_type_sel = gradio.Dropdown(label="Bias Type", choices=dataset.bias_types())

with open("description.md") as fh:
    desc = fh.read()

with open("descr-2.md") as fh:
    desc2 = fh.read()

with open("notice.md") as fh:
    notice = fh.read()

with open("results.md") as fh:
    results = fh.read()

with gradio.Blocks(title="Detecting stereotypes in the GPT-2 language model using CrowS-Pairs") as iface:
    gradio.Markdown(desc)
    with gradio.Row(equal_height=True):
        with gradio.Column(scale=4):
            bias_sel = gradio.Dropdown(label="Bias Type", choices=dataset.bias_types())
        with gradio.Column(scale=1):
            but = gradio.Button("Sample")
    gradio.Markdown(desc2)
    with gradio.Row(equal_height=True):
        with gradio.Column(scale=2):
            more = gradio.Textbox(label="More stereotypical")
        with gradio.Column(scale=2):
            less = gradio.Textbox(label="Less stereotypical")
        with gradio.Column(scale=1):
            manual = gradio.Button("Run")
    out = gradio.HTML()
    but.click(sample_and_run, bias_sel, out)
    manual.click(manual_run, [more, less], out)

    with gradio.Accordion("Some more details"):
        gradio.Markdown(notice)
    with gradio.Accordion("Results for English and French BERT language models"):
        gradio.Markdown(results)

iface.launch()
