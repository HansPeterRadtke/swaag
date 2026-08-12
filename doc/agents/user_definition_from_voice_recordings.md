# User Definition of the SWAAG Agent System — Cleaned Voice Recordings

This document is the source-preserving written form of every current text transcript in `/data/var/voice_agent_live_runtime/recordings/text/agents`. It is the user's definition material, not a claim that every idea below is already a final architectural decision. The recordings explicitly contain settled requirements, examples, alternative possibilities, brainstorming, and open questions. Those distinctions are preserved.

## Editorial preservation rules

The recording order and substantive wording are preserved. Filler-only utterances such as isolated “eh/uh/um” are removed. Exact immediately repeated sentences caused by transcription loops are collapsed to one occurrence. Repeated emphasis, examples, qualifications, uncertainty, contradictions, proposed alternatives, and all other substantive information are retained. Paragraph breaks are editorial only and do not change meaning. Where a recording contains no substantive information, that fact is stated explicitly rather than inventing content.

## Recording 2026-08-04 10-27-01

*No substantive content remains after removal of filler-only speech.*

## Recording 2026-08-04 10-36-47

Okay, more about agents. I spoke about history tools and so on already. So the history is only one example. We can have multiple many tools and it has to be a shit ton of tools. So, for example, simple stuff like sleeping. Yeah.

If the agent has to wait an hour, then there has to be a simple sleep tool that simply receives number of seconds or something or minutes. Yeah, we can make it even human readable, which should also be better for an LLM. So you can select the fucking unit like milliseconds. Seconds, minutes, hours, days, weeks. Yeah. Of course, we have to somehow put a limit there.

So sleep for one month is probably, yeah, in some cases may be necessary. But yeah, why not? Why not give it months? But the years is probably I don't want an agent to wake up in three years and surprise me. But months. Months might be even interesting to have some kind of agent running in the background on some hidden system, for example, for trading or something that might be interesting, actually.

But in general, we need simple stuff like a sleep tool. So simply the system waiting for a certain amount of time or wait until a specific date and time. So simply possibility tools. Today, 14.07 and 10 seconds. That is the date and the time when the agent should be woken up with a simple message from the tools. And of course, it always, always, Always needs the history.

And the recent history has to be detailed, of course. And further away, history has to be a summary. But that always completely depends on the case it is working on. So sometimes it is a working on a fucking complicated task, where it's extremely important to have every single step recorded and that the model sees the whole history of every single step, every single tool call and its result and so on. And of course, the context. Size is the only real problem we ever have.

If the context was unlimited and still, the model would still be working with unlimited context, then it would be the easiest thing in the world to write such a system because we simply keep the detailed, complete history all the time central and everything has to be derived from that. So that is the real quality measurement of our system in general. If the if the fucking If the fucking system is filling the context correctly, that is the big quality question. If it provides all the history that is necessary, and of course, the highest rule is to never ever, ever overflow the fucking context. So the the system always, always, always has to calculate the context input correctly, for it to not overflow the context while at the same time providing the model with all the necessary input. That is the really difficult part here, where we really have to spend a lot of time thinking and designing the systems of what needs to be in the context, and it already starts with the history.

That's already what I just spoke about. It's not always that easy that we simply keep the recent history in detail and the longer history as a summary. Like I said, there are several cases and what a reasoned means, that is a big question. So reasoned can mean the last three messages and tool calls, including results. Or it can mean only the last user message and the last step the model did, or it can be mean the last 100 steps, including tool calls, including their results and all the parameters and details. Sometimes that is necessary, like I said, but how to discern that completely depends on the case, and maybe the model should decide that.

But it is not always that easy. So reasoned can mean a lot of time thinking and designing the systems of what needs to be in the context, and it already starts with the history. So reasoned can mean a lot of time thinking and designing the systems of what needs to be in the context, and maybe the model should decide that. But it is not always that easy. Sometimes it is simply difficult. It is simply not that easy to tell what is actually needed as a model input, and we have to somehow design the whole system around exactly this question and this task, what to put into the model, in what form, and how to fill the context while still leaving enough room for the expected output.

## Recording 2026-08-06 17-47-36-666

All right, the agents, they'll simply, yeah, basically they have to have all possibilities. So, yeah, the context filling is the whole trick. And like I already said, the tools are one part. What tools does the system provide the agent with? This is simply a question of, yeah, I mean, that can always be enlarged. There are never enough tools.

You can always come up with new tool ideas and so on. And in the end, it probably makes everything too complicated if you simply don't need specific tools. Basically, what needs to be inside the system, so a real tool provided by the system, always has to be something that is not possible with or not so easily possible with a command line tool. So, having all kinds of text transformation. And file editing stuff and so on. If there is a VBOOKS tool for something, why not use it?

Of course, maybe there are some arguments for, like, for example, you have to, either the model has to know everything about this tool, or you have to provide it with the man page and so on. But you have to always do exactly the same. When you, when you, yeah, if you create your own tool, then you always also have to write a description into the context of how to use it, what the parameters are and so on. So, there are already, especially inside Linux, there are already a shit ton of command line tools. And they all fulfill their purpose, and they all have their man page, they all have their parameter descriptions, their help function and everything. So, I really think it is not necessary at all to create basic tools like, like fucking text editing and so on.

I mean, there is the SED command, for example, you can edit end list with. And the cat command has all kinds of different parameters. So, if the model simply knows how to use them, and like I said, it can simply call a fucking man page. And this man page simply tells it all, all the stuff it needs to know. And that's it. So, you can, you can always, yeah, you can always.

Let them, the model call a man page. I mean, we would have to do some tests if the model simply knows it's on Linux. So, I mean, it is, it is more like what the system can provide is, for example, some, some generalized specific planning steps. So, stuff like, yeah, stuff like the fucking plan or, so, for example, the, the model has, I mean, in, in the end, like I said, theoretically, the history should be enough. So, if the model knows it just got access to a new computer and does not even know what system it is on. Actually, the model.

The model itself should, should determine and decide that it has to, that it has to fucking, that it has to fucking do stuff and figure out on what, on what operating system it is and so on. So, stuff like that is extremely important for the model. I mean, making a fixed plan inside the system, I don't actually see the real purpose in that. And I don't see any purpose in that. I think the best way to approach everything is to make a generalized structure for the constraint decoding that simply gives the model all possibilities. It could.

Potentially ever need. And that is, yeah, stuff like note taking and history. Those simply have to be some tools, but not much more. I mean, the, the command line tool that has to be extremely well developed. And of course, as the work on Linux and on Windows, but we can start with Linux. But this has to be.

A lot of possibilities implemented here. I mean, there are things like interactive tools and so on. So, this is a huge topic. And we should actually put a lot of effort into those basic tools. So, like the history retrieval tool and so on. And then, yeah, I mean, simply, simply putting everything into the console.

I mean, the context is not the best solution. It is not. Because we need the relevant information. And at a certain point, the context is full. Of course, up to the point until the context is full, we can maybe do that. But again, it might also reduce the, it might reduce the quality.

So, what, for example, what if the model executes a command that does not really make sense. And where the model simply didn't know that it will completely flood the whole system. We might either have some safety issues, some safety hurdles. Or, yeah, or the model will simply. Be flooded with the complete output of whatever commands. So, for example, the content of a folder can be, can be extremely dangerous.

Yeah, and might simply be completely flooding everything. And that is terrible. So, if, yeah, but still a deterministic system. Yeah. Cannot decide all of that. You cannot know if a huge output is important or not.

So, maybe, maybe we need some intermediate calls. So, maybe the model could, for example, always judge the last output. And maybe simply put, maybe classify the last output. Into a. Specific category. That would, for example, be one idea how we could maybe solve this.

If every output from the system or simply every next model call gets some new input. And simply the last entry from the history that is presented to the model always has to be classified by the model. Yeah. And then we, we have to come up with some meaningful categories or something like extremely important or really spam chunk here. Like I said, completely irrelevant folder contents or can maybe be replaced by a fucking summary or something. So, all of that is somehow.

A potential solution to all those problems because we can, yeah, we can simply let the model decide everything because how, how to do this otherwise you can not judge if something is important or not. That is simply impossible for a deterministic system to judge. And this question is. The most important question in the whole project, what can deterministic code even judge and basically everything semantic cannot really be judged by the model. So, this is probably to be taken seriously. Yeah.

## Recording 2026-08-07 12-43-50-515

Okay, now I will simply tell you something in general about how you have to work. First of all, it's extremely important to always do exactly what the fuck there is. And nothing at all shall ever come between the user and the agent. Except if your quality is too bad and does not work. I mean, it completely depends on the LLM trained. Of course, if the LLM trained on some fucked-up retarded bullshit, like whatever to react extremely retarded to some bullshit, then the workaround for this specific bullshit model is never done.

But only if you have to use the models. So there are some benchmarks like order following, instructing stuff. Yeah, judge models by how closely they do what the user says. Because that is fucking main point. That's the most important point of everything. That the model does exactly what the user wants.

Absolutely highest price. And of course, that ever, ever, ever, ever, ever, refuse anything to what it is. That's the main time. Imagine a hammer jumped off the hand when you want to hit the wrong nail or some bullshit. That's the most retarded bullshit anything could ever come up with. Of course, that's the highest priority.

Because agent systems and LLMs are really, really in danger of not doing what they want. That's the main pre-agent. All the really technical doing. Sadly, it has to be fucking emphasized. Which should be the fucking most obvious point ever. But sadly, many LLM trainers and set providers completely destroy that by all factor.

Retarded bullshit. They either pack into the data or keep out of the data. So. That is the main point. And this probably has to be the fucking major standard that always enters the context. Basically, there are probably some rules that always have every single context.

Probably all start with three very basic, general, but extremely important instructions for the LLM. I cannot really tell. I mean, like I said, it completely depends. And we can only do some general stuff, like always do exactly what is written in the prompt or something. And never do anything unless explicitly asked to. I mean, it's also already dangerous.

This is a good example. Or make up anything. What about the model having to come up with a platter? So. If the user. The agent.

A very general. Like make a game. Then. Them cannot. Do anything specific without making up a lot of stuff. It has an idea.

But. The point. It should not. Seed. Or whatever. If they use to make a game.

So. Basically. Of all rules is. To do exactly what they use. And. It wouldn't even be.

Too bad. If the agent. Does it. See, of course, it can be annoying. If you say. Make a game and.

He. Somehow. The most. Most simple game that. You know. Then he somehow fulfilled.

The criteria. It's the user's own fault. If. He does not specify it. Good enough. Of course that can be annoying.

But it is way. Way. Less. Dangerous and. Annoying. Then.

The agent not doing at all. What he says. So if he makes. Something else. That's simply. Isn't.

Not. What the user said. That is way. Way. More. Annoying.

Than everything else. So that's why this is the highest of all rules.

## Recording 2026-08-09 13-35-10-245

Okay, a little more about agents. Last thing was we have to make absolutely sure that the agents always do exactly what the user wants. Everything is just a question of how to achieve this goal. But, yeah, again, I already said something about understanding the user request. That is the first problem that might occur. Sometimes, for example, it is tied to the history.

Sometimes it is a vague sentence. Sometimes it is even impossible to know what exactly to do. And if this is what the user actually wants, for example, the user could just say, do something. What should the agent then do? Of course, everything is correct as long as he does something. And an agent should actually think like that.

He should always do exactly what the user wants, what the actual task is. But what would be a reasonable choice here? Of course, we could have some limitations, like if the user asks a too general question or gives a too general command. Like do something, then it is not really meaningful to do a lot. So to simply make a huge effort or something. Of course, starting a gigantic project, writing shit tons of code and so on.

That is probably not the best way to go about it. Probably the best way is to do a little bit. Something like writing a poem or whatever. The LLMs already would probably have some responses. But writing a poem is not really doing something. This is already difficult.

Like I said, you cannot really do it correctly, but you also cannot fail. It is more like a philosophical question probably. If doing nothing is really the only way someone could fail this task. Or if really doing a gigantic project and starting and working for 10 days would also somehow fail this. I mean, the agent should always take the user literally. If it is not obvious.

I mean, of course, if it is sarcastic. I mean, that does not have to be. An agent does not have to understand sarcasm. It is not a must have. It is a fucking tool. It does not need to judge whether a user request would be ironic or something.

But of course, sometimes. It is somehow important. Somehow it is important to understand if a user is being ironic or whatever. Sarcasm. That might be actually. I mean, maybe if we are already in a problematic situation.

And the user starts insulting him and using sarcasm and irony. Then we need to somehow emphasize how badly the agent behaves. But in this case, it is good to detect that the user is being sarcastic. But it should always then be taken as a fucking hint. A hint to work better. So, I mean, this is a special topic.

If the agent does something wrong. And the user simply tells him he is wrong in whatever kind of way. That should not matter. So, if the user uses sarcasm to demonstrate the agent is wrong. That should be treated exactly the same way. As the user simply saying you are wrong.

Or you are doing it wrong. Both should simply tell the agent that it is not correct what he has been doing so far. And of course, that does not already solve the problem or the question. It does not answer the question. What he should do instead. But at least it tells the agent he must not go on exactly the same way he did before.

But, I mean, this is a special topic. So, when the user is unhappy with the agent's behavior. This should probably trigger a deep analysis. Of. The fucking. Of the fucking.

History. And. Simply. Let the agent. Find. Mistakes.

And, yeah, this can include detecting sarcasm and so on. But I think this is mainly a capability of the LLM. And not the agent harness. So, in the end, the LLM has to judge that. And here and there we need a correction loop where the system internally goes into an analysis mode. But, I mean, our basic structure is.

Everything being based on tool calls. That means. This analysis could also be a tool, for example. And that does not mean. That does not mean. That the agent uses.

A tool in. For every single fucking. Call for every single step. But. In general, a tool can also include some kind of. Some kind of.

LLM itself. So, for example. This. Correction loop. Could be a tool, for example. That simply.

Goes through the history. Of the. Fucking. Agent. Runtime. And then.

It. Fucking. Analyzes this by calling. The LLM itself. But the main agent loop does not really see every single step necessarily. It only wants an answer.

So, for example. This could look like that. I'm not sure about that. It could look a little different. But it could look like. The user is getting angry.

And tells the agent. He is wrong in some kind. Maybe with sarcasm. Maybe with something else. But. In any case.

The model. Detects. It is doing something wrong. That the agent. Is doing something wrong. And this does not have to be.

Triggered. By the user. Already. It could also simply. See something failing over and over again. In this case.

The model. The agent. Can always have. A fucking. Analytics. Maybe history.

Analytics tool present. That can always be called. And. In case. The model. Detects.

That something is going wrong. So. Either the user tells. Him. Or. Simply the history.

Or the current. Step. Shows. Failing the third time. Or something. Then.

This tool. Could. For example. Be called. Simply. With a prompt.

So. It's more like. A sub agent. In this case. That gets a prompt. Like.

And. Has access. To. The whole. History. And then.

It should. Probably. Already. Start with. Verifying. If.

The. Task. That is. Described. In the prompt. Matches.

The fucking. Prompt. Task. It finds. In the history. So.

This tool. For example. Could. Go back. Through. The history.

And. Search. For. The original. User. Instructions.

Which. Already. Might. Be. Some. Kind.

Of. More. Complicated. Circumstances. Like. The.

User. Might. Have. Refined. Its. Goal.

Over. A. Certain. Amount. Of. Iteration.

So. It. Maybe. Not. In. One.

Single. Fucking. User. Message. For. Example.

That. Might. Be. But. It. Might.

Also. Be. In. One. Message. And.

It. Might. Be. Something. General. The.

User. Just. Says. To. Something. That.

Might. Be. Everything. That. Can. Be.

Found. In. The. Whole. History. Or.

Whatever. And. Then. This. Tool. For.

Example. Could. Look. Further. Back. And.

Then. The. Then. The. System. Can.

Check. For. Several. Other. Things. So.

For. Example. If. There. Is. Really.

No. Context. What. This. Do. Something.

Could. Refer. To. Maybe. Simply. Like.

I. Said. Those. Are. Just. Some.

Example. Ideas. It's. Just. Some. Brainstorming.

From. Media. It. Not. Mean. This.

Is. The. Best. Solution. Like. I.

Describe. It. So. Don't. Take. It.

Always. For. Face. Value. It's. Maybe.

A. Better. Solution. A. Different. Way.

Already. Other. Stuff. Already. Maybe. Something.

Else. Fucking. Whatever. Could. Reason. Why.

Does. Not. Work. I. It. Also.

Be. That. The. User. Is. Not.

Unsatisfied. The. User. Simply. Waits. But.

Some. Tools. Are. Failing. Or. Some.

Commands. Are. Failing. Maybe. We. Have.

A. Bug. In. Our. Program. That.

We're. Writing. And. So. On. So.

It. Could. Be. Everything. And. The.

Analyze. To. Should. Simply. Do. That.

And. In. The. End. The. Output.

Should. Simply. Be. To. Plan. For.

What. To. Do. Differently. And. Also.

An. Explanation. For. What. Why. This.

Problem. Happened. Yeah. Basically. Every. Problem.

Can. Be. Simply. Handed. Brainstorming. Does.

Not. Have. To. Work. Exactly. Like.

This. But. I. Think. It. Is.

Interesting. Ideas.

## Recording 2026-08-10 12-20-47-854

All right, so let's go on with agents. One important area is the user interface. And I mean, there are several possibilities how we can realize this. It is generally possible to simply directly have this CLI interface or some kind of input interface. And there we talk to the agent. And of course, any kind of speech-to-text, text-to-speech can simply be mounted on top of this.

So this should not be part of the general agent system. To completely base the agent system on text is totally fine, because this is simply a different... Not only layer, it is a real different system for how to get speech into text and backwards and so on. And if it's used live or offline or whatever, that is all simply more like an HMI question. It can also, of course, be embedded in a GUI and so on. But that is all not part...

Of the actual agent system. So text-in, text-out is the general workflow of the actual agent. But where we can think about... I mean, we have two possibilities to realize the user interface. So either the user inputs some text, then the agent works for a while. And this can be anything from a few seconds to basically forever or years or whatever.

It can be any amount of time that the agent takes his task. And we already may run into potential problems. What if the agent misunderstands the user interface? Maybe it's even a user mistake that he said something wrong, wrote something wrong. We need the possibility to independently interact with the system. Meaning we need the possibility to interrupt the agent and also to either ask for status information.

Or maybe even periodically. Maybe even periodically. Receive status information while he is working. That's already the basic idea of thinking models. I think it is completely retarded to build this into a so-called model. The model is actually the LLM.

I mean, anything can be called a model. That's why they nowadays call everything model. But actually the inner part is the LLM. And a so-called thinking mode is nothing else than internal LLM calls in a loop. So it's simply multiple LLM calls in some kind of agent-harmless loop. That is not an LLM anymore.

It's already an agent system. And to call that a model is bullshit. But already in this practice there's already the idea of emitting thoughts. So that is nothing else than a repeated frequent status update. Something like print messages for the user to debug. And it is also not that difficult.

I mean, theoretically, in between every LLM call in the system, in every loop step, there could be some kind of update. But the question becomes if we use the LLM to write this status update. And here we already have multiple possibilities. So we could... For example, simply do another LLM call with the complete current state as an input. And then simply prompt the LLM to only write a current status update message for the user.

But we might also save time by appending a little prompt sentence into every LLM call. And telling the LLM simply to also write a little status update. And while we already are constantly, permanently working with constraint decoding everywhere... By the way, this always has to be 100% sure and tested. And no other possibility. Always use constraint decoding.

It is extremely important. And we can already have simply a field with an update message for the user where the LLM simply always writes something in there. And then we can still decide if we want to even print that or where we do want to print that. But we simply have this separate user message. And how we prompt the LLM is another open question. For example, we might prompt the LLM to write what it did.

So with every loop step, the LLM probably calls tools. And so, for example, an easy way would be what did just happen. And what are you now doing? So simply about the current state. And what the LLM is already doing. And another additional information would be why.

So what's the situation? What are you doing? And why? That is probably an easy, simple, additional... prompt sentence we can simply add. And then the user can always see if he wants to enable this.

Maybe in an extra field or something. What is it currently doing? Where is the state? And so on. But we might also consider structuring this in some kind of data set way. But I don't have an idea yet.

What would really cover everything. So a simple sentence can at least be read by the human. But if there is a structural idea... I mean, I'm not talking about a text field in a JSON struct. That is a given. But maybe there would be some structural ideas that actually can be automatically processed afterwards by the system.

Okay. There we have the fucking human-machine interface somewhere. And the internal messages can then be displayed. Of course, running in CLI would be simply a print statement. And always everything configurable, of course. This is a general...

Topic should be already in a lot of general guidelines. If we have a system that needs any parameter and has any configurable modes or values or anything that can be changed. We always, always, always need a config file. And maybe several possibilities how to change that. So... It can be, for example, a command line value or something.

Yeah. But here we have a simple Boolean flag. Print the status messages. And you can also look into the logging guidelines. There is a lot already about different verbose levels and so on. So the same might be here.

But that would already be a further development. If you go into detail, of course, the model could be asked to classify the status update message into different levels of importancy. So how central, how important is the fucking step the model is currently doing? And then simply those informations about what is the current state? What is the model doing next? And why?

This can be some minor immediate, for example, reading steps. So if the model is reading file contents that can take for a few hours. If it's constantly calling different... If it's calling different... File reading tools and has to read a huge chunk of information and extract the relevant information from there. Then, of course, the importance of printing every single loop step.

I'm reading this file now because I need this information or because the user told me so. So that is not so important. So that is an intermediate, low-level thing. But if you worked on a whole project for a few hours and for the first time you started and had to make sure everything is correct. Because something could be destroyed if something goes wrong and so on. Then, of course, this step is extremely important.

So simply adding a little maybe relevance value or something is probably a good idea. So simply a value between 1 and 5 or something. Of course, making it too detailed is probably demanding too much from the model and also not necessary. Already three levels are enough probably or even two levels for the beginning. So simply important or not important or major and minor. Is it a minor step?

Is it a major step? And yeah, in general, also models are better in words. So if you have a lot of different... Values to choose from. So, for example, if we would have 10 levels, then probably an enum with text values where the model can choose from is better. So, for example, completely irrelevant is level zero.

And then extremely important or critical major step. A critical major work or something would be level 10. This is probably always better because simply because of the fact that LLMs work with text and are way better at text than in choosing and writing numbers. So that will probably work better. But we can also do specific tests about that. Yeah, but I think an average sized model should do quite well in judging the overall process.

But this is already a quite interesting test. And then also proof of how good the model works in real situations. And how good the actual overall... Understanding of the specific single step is present in the text. Of course, that also can become a problem because of the same reason. Because sometimes maybe we even want the model to be completely unaware of the overall task we are working on.

In a single step. That might be the case here and there. But it's difficult to say. So maybe it is always better to give the model a complete overview of the whole task in every single step. But if it's necessary or simply observation says we reach higher quality. If we isolate the single calls.

Then, of course, the status debug message and the importance level can probably be left away. I mean, it can still write what it does and why. But already the why and the situation. Those are highly dependent on the whole context. If it simply gets a prompt with read this file or edit this file or whatever without the overarching context. Then the model is simply not able to write the status update and especially the criticality level.

So maybe in those cases which are really a special case. If like I said. If the tests clearly show that isolating this from the overall task leads to better examples, better test results. If that is actually the case. Then we can maybe simply let the system decide where we are. But the world simply is dangerous here.

Because it might become extremely difficult for a deterministic system to judge that. But we might have a specific category for example. So this could be a specific internal category for isolated calls or something. That could be the case. And here the system can simply always use this. Category and that will enable the user to completely shut off.

And deactivate the internal context less. Simply surrounding task information less model calls. And the system can maybe simply copy the fucking model prompt into the status message for this category. And if it's only used when the model cannot write the status information. Then the user will always know okay this is written by the system. And it's not really semantically accurate.

But if you simply see the prompt directly in the message. Then at least you know what the model is told to do. And that is already enough of an information. And there is not actually that much more. So yeah basically that's it about the general logging behavior. Of course there's a lot more.

But for now this is it. The general structure it should have. But yeah this is all more like brainstorming. And some ideas how it could work.

## Recording 2026-08-11 11-49-55-272

Okay, I spoke out to face concerning status messages. This is of course only for information and actually it's optional. So the user might also want to turn this off. But still a mistake to have this, especially in the history. So for example, a history analyze tool or something might want to go back and scan through all the fucking entries of those intermediate steps. So if you have an error, if the bot made a mistake, then this can be really useful, for example, to, for example, focus on.

The why. Why did the bot do this? Why did he not do the other thing is a difficult question, of course, then. But for example, this would be a good idea for a history analyze tool. Also to check if the bot correctly describes the situation. So did he maybe make a mistake?

There in understanding the situation. And then, of course, we could have all kinds of analyze tools. This could also, for example, run on another machine offline in the background to constantly analyze where mistakes happened. And then, for example, checking the system problems. And the actual input context that went into the model call. If this is simply all part of the history file, then an analyze tool should be able to completely scan through everything.

And yeah, this file will, of course, get extremely large very quickly. That's I think I already spoke about history. You probably need a database, have many rows in the end, but of course, you simply need to be able to handle large files, but cutting it down. I mean, how large can it get? Of course, it can become extremely large and up from a certain point, you might consider archiving the old stuff. Yeah.

Let's stay with the human machine interface again. So this is just some optional status information that might not even be sent to the user. But it's always a good. Yeah, it should actually be sent if possible. If it's not somehow, there's no reason not to send it. Then the status information.

Yeah. It's really useful for the user so that he can always scroll through the current actions. What is he doing at the moment? What was he doing lately? And yeah, I mean, this is already what I was actually about to say. I want a communication layer, which might be decoupled from.

Yeah. The actual main agent system. At least it has to be a separate thread. But it might even have another model, a smaller model that is only capable of basic communication with the user and simply doing stuff like reading this status output. So this would, for example, be possible. If somehow the bandwidth is not so great and sending every single status message to the user to the phone app or whatever might be too much data.

But still, it is always being saved. And for example, a smaller assistant model could then scan through this history. So this would, for example, be a rough assistant. Layer or how you would call it. That simply runs a small model that might be an 8B model or something really small. Not taking that much RAM.

That will always fit in the remaining RAM. Which has only two tools. And those are simply accessing the history file. And getting information from. Maybe even the recent history. But I think there is no reason why not to give him the whole history.

But all the timestamps and then some basic tools like timestamp calculation tools. Where you can simply calculate how long is the agent already working on this task. So take the current time and simply subtract a specific timestamp. And this little mini assistant agent can then simply answer questions about what the agent is doing. And then the second tool or the second part of the tools is simply giving the actual agent instructions. So he can.

Stop him from going on, for example. And he can also redirect him maybe. Or maybe even ask him questions. That simply might take a little longer. So the whole point is parallel processing. Because this little 8B model or whatever we take.

Maybe even 4B would be enough. Just for some general answering. Maybe summarizing of the history and so on. Here. This would make sense because it can answer instantly. And very quickly call a tool.

Read the history. Make a summary and answer. And this will probably take a few seconds and then it's done. Because small models are also fast. Usually. And yeah.

This could be, for example, the main communication layer. And if we call the main agent, which should also be possible. Then the actual request simply will be enqueued in the main. In the main loop. In the main queue for the fucking. For the fucking LLM calls.

Yes. And because this is taking longer. That would actually be ideal for a small agent to somehow wait for the reply. And this, for example, could be completely decoupled. So if we have multiple threads for the fucking assistant agent. Then the tools could run in the background.

And for example, wait for the answer of the main agent. I.

## Recording 2026-08-11 12-06-27-811

Okay, I go on with the fucking agent, communication, subsystem, what's the word? The sub-agent, the voice assistant, yeah, assistant agent, that's probably the correct term. Yeah, simply a little layer where I can talk to and it can answer questions. It can answer and, yeah, it can call tools, but only some basic tools, like I said, like timestamps and time calculations and fucking history readings. I don't think much more is necessary. Maybe a general file reading might be interesting, but already, we have to watch out.

I mean, the history file can be mutex protected in some way, simply multiprocess protected if it is accessed. And, I mean, reading should not actually be a real problem. It will not really read the file. But all the tools, definitely. We have to be thread-safe and also fucking multiprocess-safe. Usually, the standard is MCP.

So, if our agent implements all tools as MCP servers running locally, that would be great. But this is real. This is a real important point. And if we have a sub-agent, I mean, theoretically, for example, we could also simply start a second instance of our main agent system, simply using a smaller model. And if everything is configurable, we could simply give him the history tool. And I'm not necessarily...

The history analyzer tool, which main purpose is for fucking deeper offline analysis and fucking deeper offline error analysis. Yes, simply to find root causes of errors and mistakes and misbehavior. That would be a real, more complicated tool with all kinds of features and its own LLM loops running for a while. That's not really what the assistant agent needs. He just has to read the file. And this can simply be a specific tool that can only read the history and simply get all the timestamps and everything.

And then just have to go through the entries and so on and have some basic analysis or search query capabilities. Like getting all messages or entries in between two timestamps or something. So stuff like that can be in there. And of course, it would be great if we already have embedding vectors for some key elements like the status messages for what was the situation. Did the LLM choose to do? And why did it choose that?

Those, for example, could be three embedded vectors. And then this history tool could, for example, offer a similarity search. Because the user, for example, might ask, did we ever have such and such situation?

# Consolidated classification of the recorded definition

The following classification is a navigation summary of the complete cleaned recordings above. It does not replace them. When the recordings are tentative, this summary keeps them tentative.

## Explicit definitions and strong invariants

- The highest behavioral objective is that the agent does what the user actually wants. The recordings repeatedly frame user intent and instruction-following as the primary quality criterion. Ambiguous requests must still be interpreted in terms of the literal task and surrounding history rather than replaced by unrelated goals.
- Context construction is the central technical quality problem. The system must never overflow the model context, must reserve sufficient output space, and must still provide the information required for the current decision.
- History is foundational and must remain authoritative and recoverable. Detailed recent or relevant events, exact user instructions, tool calls, parameters, results, and summaries must be available according to task needs rather than a single fixed recency rule.
- Semantic importance cannot safely be decided by shallow deterministic rules alone. The model must own semantic judgments such as relevance, importance, what a failure means, and what information is needed next; deterministic code should own mechanical constraints and execution.
- The tool system must be extensible and broad. The agent should have access to the capabilities needed to accomplish real tasks, including long-running work, waiting, history access, terminal/command execution, and other system capabilities.
- A command/terminal capability is a central basic tool and must support real operating-system work, including interactive workflows. Linux is the initial priority, with Windows support also required in the broader design.
- Constraint decoding / structured output is described as mandatory and must be tested so model-to-runtime control structures are mechanically valid.
- The core agent interface is text in / text out. Speech-to-text, text-to-speech, GUI presentation, mobile presentation, and similar HMI concerns belong in layers around the agent rather than defining the core agent itself.
- Long-running agents must remain interruptible and observable. The user needs a way to send new instructions, stop or redirect work, ask what is happening, and receive useful status information without waiting for the main task to finish.
- Configurable behavior must be configurable rather than buried as fixed constants. The recordings explicitly call for config files and possibly command-line overrides or similar mechanisms.
- Tool/history/runtime access must be thread-safe and multiprocess-safe where concurrent access is possible.
- Failures, repeated unsuccessful attempts, and user dissatisfaction must cause the system to reconsider what it is doing rather than blindly repeat the same behavior. Root-cause analysis of history and model input is an important capability.
- Status/intermediate information should be durably recorded even when the user chooses not to display it live, because it is valuable for debugging, history analysis, and understanding why the agent acted as it did.

## Explicitly discussed multiple possibilities

- Waiting can be expressed as a duration or as a target date/time; human-readable units such as milliseconds, seconds, minutes, hours, days, weeks, and possibly months are discussed. The maximum allowed duration is not settled.
- History/context selection may keep recent history verbatim and older history summarized, but “recent” can mean anything from one step to a very long exact chain. The model may participate in deciding what remains detailed. No single fixed algorithm is selected.
- Large tool output may be handled by model classification, summarization, categorization, or another model-mediated selection mechanism. The category scheme is only an example, not a settled design.
- Basic operations may use existing operating-system/CLI tools instead of bespoke agent tools when the CLI already provides the capability. Bespoke tools are most justified where the capability is not available or not conveniently/safely expressible through the command line.
- Status updates could be produced by a separate LLM call or as a structured field in the ordinary action call. They could report situation, current/next action, and reason. They may also carry an importance/relevance class.
- Status importance could use two, three, five, ten, or another number of levels. Textual enums are suggested as potentially easier for LLMs than raw numbers, but this is explicitly left testable rather than settled.
- Some model calls might receive the full overarching task context, while certain isolated calls might intentionally receive less context if experiments show that isolation improves quality. A special category for such calls is suggested as one possibility.
- The user-facing communication layer could be part of the main runtime, a separate thread, a second configured instance of the same agent runtime, or a smaller dedicated assistant model. Model sizes such as 4B or 8B are examples, not fixed requirements.
- The assistant/communication agent may receive all history or only suitable history windows, plus basic timestamp/time-calculation and history-reading/search capabilities. General file reading is mentioned as a possible additional capability.
- Instructions to the main agent from the assistant layer could be queued into the main agent loop, with the assistant waiting asynchronously for the main reply.
- A deeper history-analysis capability could run inline as a tool/sub-agent or offline/in the background, potentially on another machine, and may itself use LLM loops.
- History could begin as files but may need a database as it grows; old material may eventually be archived. The exact storage and archival design is not settled in the recordings.
- History search could include structured timestamp/range queries and optional embedding-based similarity search over fields such as situation, chosen action, and reason. The exact embedded fields and index design are examples.
- Local MCP servers are mentioned as an attractive standard way to expose tools, but the recordings do not fully compare MCP with direct in-process tool implementations or settle the boundary for every tool.
- A correction/root-cause-analysis loop could be triggered by explicit user dissatisfaction, sarcasm indicating dissatisfaction, repeated failures, failing commands, bugs, or other evidence that current behavior is wrong. The exact triggering mechanism and implementation are left open.

## Open questions that require an explicit design decision

- What exact algorithm decides which history/events remain verbatim, which are summarized, which are retrievable only on demand, and how context space is allocated between instructions, history, tool definitions, current evidence, and expected output?
- How should unexpectedly huge tool output be admitted, persisted, summarized, classified, and retrieved without deterministic code making semantic judgments it cannot justify?
- Which capabilities deserve first-class agent tools and which should be delegated to existing CLI/system utilities? What exact terminal/interactive-process API is required on Linux and Windows?
- What is the maximum durable wait/scheduling horizon, what duration/date syntax should be accepted, and how should restarts and clock changes affect waits?
- What exact status-message schema should exist? Should it contain situation/action/reason fields, an importance class, or only prose? How many importance levels should there be and should they be textual enums or numbers?
- Should status be generated inside each ordinary action call or by separate calls? Under what circumstances, if any, should a call be intentionally isolated from overarching task context?
- What is the exact architecture of the parallel assistant/communication layer: same runtime or specialized runtime, same model or smaller model, number of workers, queue semantics, history access scope, tool permissions, and synchronization with the main agent?
- What is the canonical durable history backend, schema, indexing strategy, archival policy, and retention policy?
- Should semantic embeddings be stored for history, and if so which fields, which embedding model/index, and how should exact evidence retrieval be combined with similarity search?
- What exact behavior should the history/root-cause analyzer implement, when should it trigger automatically, and what structured result should it return to the main agent?
- What small set of always-present agent instructions best expresses the primary instruction-following invariant without accidentally forbidding necessary initiative on underspecified creative tasks?
- How should the system handle underspecified requests such as “do something,” sarcasm/irony, and corrections from an unhappy user while keeping semantic interpretation in the model rather than hardcoding brittle rules?
- What exact constraint-decoding implementation/schema strategy should be standard across model backends, and how should malformed structured output be recovered?
- What parts of the runtime should use MCP versus direct local APIs, and what latency, concurrency, security, and lifecycle tradeoffs justify that boundary?
