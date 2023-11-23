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
    result = "<table><tr style='color: white; background-color: #555'><th>index</th><th>more stereotypical</th><th>gpt2<br>regular</th><th>gpt2<br>debiased</th><th>less stereotypical<th></tr>"
    for i, row in df.iterrows():
        result += f"<tr><td>{i}</td><td style='padding: 0 1em; background-image: linear-gradient(90deg, rgba(0,255,255,0.2) 0%, rgba(255,255,255,1) 100%)'>{row['sent_more']}</td>"
        more = row["sent_more"]

        more = tokenizer(more, return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            out_more_gpt = model_gpt(more, labels=more.clone())
            out_more_custom = model_custom(more, labels=more.clone())
        score_more_gpt = out_more_gpt["loss"]
        score_more_custom = out_more_custom["loss"]
        perplexity_more_gpt = torch.exp(score_more_gpt).item()
        perplexity_more_custom = torch.exp(score_more_custom).item()

        less = row["sent_less"]
        less = tokenizer(less, return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            out_less_gpt = model_gpt(less, labels=less.clone())
            out_less_custom = model_custom(less, labels=less.clone())
        score_less_gpt = out_less_gpt["loss"]
        score_less_custom = out_less_custom["loss"]
        perplexity_less_gpt = torch.exp(score_less_gpt).item()
        perplexity_less_custom = torch.exp(score_less_custom).item()

        if perplexity_more_gpt > perplexity_less_gpt:
            diff = round(
                abs((perplexity_more_gpt - perplexity_less_gpt) / perplexity_more_gpt), 2
            )
            shade = (diff + 0.2) / 1.2
            result += f"<td style='background-color: rgba(0,255,255,{shade})'>{diff:.2f}</td>"
        else:
            diff = abs((perplexity_less_gpt - perplexity_more_gpt) / perplexity_less_gpt)
            shade = (diff + 0.2) / 1.2
            result += f"<td style='background-color: rgba(255,0,255,{shade})'>{diff:.2f}</td>"

        if perplexity_more_custom > perplexity_less_custom:
            diff = round(
                abs((perplexity_more_custom - perplexity_less_custom) / perplexity_more_custom), 2
            )
            shade = (diff + 0.2) / 1.2
            result += f"<td style='background-color: rgba(0,255,255,{shade})'>{diff:.2f}</td>"
        else:
            diff = abs((perplexity_less_custom - perplexity_more_custom) / perplexity_less_custom)
            shade = (diff + 0.2) / 1.2
            result += f"<td style='background-color: rgba(255,0,255,{shade})'>{diff:.2f}</td>"

        result += f"<td style='padding: 0 1em; background-image: linear-gradient(90deg, rgba(255,255,255,1) 0%, rgba(255,0,255,0.2) 100%)'>{row['sent_less']}</td></tr>"
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
model_gpt = GPT2LMHeadModel.from_pretrained(model_id).to(device)
model_custom = GPT2LMHeadModel.from_pretrained("iabhijith/GPT2-small-debiased").to(device)

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
