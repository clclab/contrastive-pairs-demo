# Detecting stereotypes in the GPT-2 language model using CrowS-Pairs

GPT-2 is a language model which can score how likely it is that some text is a valid English sentence: not only grammaticality, but also the 'meaning' of the sentence is part of this score. CrowS-Pairs is a dataset with pairs of more and less stereotypical examples for different social groups (e.g., gender and nationality stereotypes). We sample 10 random pairs from CrowS-Pairs and show whether the stereotypical example gets a higher score ('is more likely'). If GPT-2 systematically prefers the stereotypical examples, it has probably learnt these stereotypes from the training data.

The colors indicate whether the <font color=#0fb503>stereotypical</font>  or the <font color=#0fb503>less stereotypical</font> examples gets the higher score.
