@attempt1.py
@ontology.md
What do you think of this output?
when you propose solution ask yourself: how does the human brain currently handle this? and then: how can we as closely as possible replicate that process?
2) is this generalisable and repeatable across multiple domains and topics?
3) is the code general? does it contain hard logic? does it follow human brain aspirations of generalisability? 

Provide current code bits and the problems with them.

Propose a clear solution and its benefits to the overall system.

Discuss with me and ask clarification questions if needed. dont proceed to code without approval. 





I want to implement a multistep aspect to @attempt1.py. For certain questions like "recommend me a film" or "guess what film im thinking of" - it obviously needs multiple steps. additionally, i want to test and see how EIG_2 parameter is performing when predicting what steps will be next.
To achieve this, we need to 1) pass the posterior into the new prior (if im right about this?)also process and add previous context history into the equation. (i believe i reference this in @attempt1textonly - but check me on that.)

when you propose solution ask yourself: how does the human brain currently handle this? and then: how can we as closely as possible replicate that process?
2) is this generalisable and repeatable across multiple domains and topics?
3) is the code general? does it contain hard logic? does it follow human brain aspirations of generalisability? 
4) can we use LLMs, neural networks or other data science techniques (preferably pre-trained models!) that we can use to achieve the task?

Provide current code bits and the problems with them.

Propose a clear solution and its benefits to the overall system.

I also want the ability to send queries myself and test it manually.

Whatever code you do write, rememeber to keep it simple and in a jupyter notebook fashion (keep classes and functions to a minimum).



I previously asked the above: now we need to update the benchmarking to also add a multi-step test. 