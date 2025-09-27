
# General guidance
Context:
@attempt1_TextOnly.py - is your source of truth for our north star and theory behind this system
@ontology.md - is our current approach to how the data is structured in the neo4j db.
@seed.cypher - is our seed data we can query.

when you propose solution ask yourself: how does the human brain currently handle this? and then: how can we as closely as possible replicate that process?
2) is this generalisable and repeatable across multiple domains and topics?
3) is the code general? does it contain hard logic? does it follow human brain aspirations of generalisability? 
4) can we use LLMs, neural networks or other data science techniques (preferably pre-trained models!) that we can use to achieve the task?

Provide current code bits and the problems with them.

Propose a clear solution and its benefits to the overall system.

Whatever code you do write, rememeber to keep it simple and in a jupyter notebook fashion (keep classes and functions to a minimum).

Discuss with me and ask clarification questions if needed. dont proceed to code without approval. Work in small incremental steps, write tests for each component/module. 

Output to review:

[SUCCESS] Started session: 26c6a45a-28a2-4cf4-8e5c-a64cfa952af1
You: hello
DEBUG: VerifyMusicRights - utterance: 'hello'
DEBUG: VerifyMusicRights - has_music_terms: False
DEBUG: ❌ VerifyMusicRights set to 0.1
VEGETA [ASK]: Could you provide more information about what you need?
Confidence: 0.0%
Reasoning: LLM analysis: extremely_vague query needs none clarification

(Provide your response to help narrow down the answer)
You: im thinking of a film. can you guess it?
DEBUG: VerifyMusicRights - utterance: 'im thinking of a film. can you guess it?'
DEBUG: VerifyMusicRights - has_music_terms: False
DEBUG: ❌ VerifyMusicRights set to 0.1
VEGETA [SEARCH]: Let me search for information about missing_facts...
Confidence: 18.0%
Reasoning: High uncertainty with multiple candidates, searching for more facts

You: find me a film with pierce brosnan
DEBUG: VerifyMusicRights - utterance: 'find me a film with pierce brosnan'
DEBUG: VerifyMusicRights - has_music_terms: False
DEBUG: ❌ VerifyMusicRights set to 0.1
VEGETA [ASK]: Are you referring to Pierce Brosnan, the actor known for playing James Bond, or someone else with the same name?
Confidence: 15.3%
Reasoning: Low confidence in Keanu Reeves, asking for specifics

(Provide your response to help narrow down the answer)
You: yes im referring to pierce brosnan. What other films did he star in?
DEBUG: VerifyMusicRights - utterance: 'yes im referring to pierce brosnan. what other films did he star in?'
DEBUG: VerifyMusicRights - has_music_terms: False
DEBUG: ❌ VerifyMusicRights set to 0.1
VEGETA [ASK]: Are you referring to Pierce Brosnan, the actor known for playing James Bond, or someone else with the same name?
Confidence: 13.9%
Reasoning: Low confidence in Keanu Reeves, asking for specifics

(Provide your response to help narrow down the answer)
You: what film did daniel craig star in?
DEBUG: VerifyMusicRights - utterance: 'what film did daniel craig star in?'
DEBUG: VerifyMusicRights - has_music_terms: False
DEBUG: ❌ VerifyMusicRights set to 0.1
VEGETA [ASK]: Are you referring to Daniel Craig as the actor who played James Bond or someone else with the same name?
Confidence: 20.0%
Reasoning: Low confidence in Casino Royale, asking for specifics

(Provide your response to help narrow down the answer)
You: yes
DEBUG: VerifyMusicRights - utterance: 'yes'
DEBUG: VerifyMusicRights - has_music_terms: False
DEBUG: ❌ VerifyMusicRights set to 0.1
VEGETA [ASK]: I'm having trouble understanding what you need. Could you provide more details about what you're trying to accomplish?
Confidence: 0.0%
Reasoning: LLM analysis: extremely_vague query needs intent clarification

(Provide your response to help narrow down the answer)
You: quit
[SUCCESS] VEGETA system closed

Questions i have:

1) why isnt our multi-turn / session state management working?
2) How does the brain currently handle small/talk hello? I assume it doesn't go through its whole inference process like we do. or perhaps, it does and is fast/doesnt need to go through the whoel thing but its confidence is high/its error rate is low? prior / g(u)-> v' is predicting "hello" as the first thing to arrive? (follow by our EIG_2 where we should be predicting hello->im thinking of a film, can you guess it?)





