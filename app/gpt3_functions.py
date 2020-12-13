
#!/usr/bin/env python

#REQUIREMENT: pip install openai

import openai


#orig_prompt= "Here is an award winning short story:"+ "\n"+ "A woman paddling on a surfboard. Maria sat on her surfboard, as it bobbed up and down on the coastal waves. The neon lights of Ocean Drive burned on the shore in the distance, like green and pink embers. It was just her, the water, and whatever starlight got past the Miami light pollution. A shark swimming in the ocean. She was paddling back to shore when she saw a shark fin in the water. At first she thought it was a manatee or a dolphin. She’d occasionally seen those swimming around here, spent countless hours watching them under the night sky. But this was different. A boat in the lake. A bright spotlight shown down on Maria. Maria looked up, and saw a small fishing boat, floating half-way between her and the shoreline. A stout man in an orange rain coat and fishing boots stood on the deck. “Get out of the water!” A gruff voice shouted. Maria's eyes went wide. She turned and saw the large dorsal fin rapidly advancing. The shark’s body hovered just under the surface like a dark cloud. Maria’s hands dove into the water, paddling faster than she’d ever paddled in her life. Her legs kicked frantically. "

prompt_file_path= './GPT3Prompts/origin_prompt.txt'
stored_story_file_path= './GPT3Prompts/stored_story.txt'

prompt_file= open(prompt_file_path,'r')



orig_prompt= prompt_file.read().strip('\n')
orig_prompt=orig_prompt.strip()

#print(orig_prompt)


story_prompt_file_path= './GPT3Prompts/story_so_far.txt'

openai.api_key = "sk-XHaEg50XHxqsa7j3aSUx7YQNGsHTgmRWjNmRXh3k" 

PROMPT_TAG= "[[Prompt]]:"
STORY_TAG= "[[Story]]:"


def new_story_with_caption(caption, temperature = 0.80):

    story_prompt_file= open(story_prompt_file_path, "w+") 
    stored_story_file= open(stored_story_file_path,'w+')

    constructed_prompt=  "Here is an award winning short story:"+ "\n"+ PROMPT_TAG+ caption + "."+'\n'
    pass_prompt = orig_prompt +"\n\n\n" + constructed_prompt 
    
    response = openai.Completion.create(engine="davinci",temperature=temperature, prompt=pass_prompt, max_tokens=330, presence_penalty =0.05, stop=['\n\n','\n\n\n',PROMPT_TAG])
    
    reply = response["choices"][0]["text"]  
    print(f"the type of the response is {type(reply)}")
    #reply = reply.decode('utf-8')
    reply=reply.strip('\n')
    reply=reply.strip()

    print(reply)
    
    story_prompt_file.write(constructed_prompt)
    story_prompt_file.write(reply)

    reply_clean= reply.strip("[[Story]]:")
    stored_story_file.write(reply_clean)

    story_prompt_file.close()
    stored_story_file.close()
    
    return reply_clean

#When user chooses to continue the story through uploading an image
def continue_story_with_caption(caption, temperature = 0.8):
 
    story_prompt_file= open(story_prompt_file_path, "r") 
    story_so_far= story_prompt_file.read().strip('\n')
    story_so_far=story_so_far.strip()

    constructed_prompt=  "\n\n"+ PROMPT_TAG+ " "+ caption + "." +"\n"

    #print("CONSTRUCTED"+ constructed_prompt)
    pass_prompt = orig_prompt +"\n\n\n" + constructed_prompt 
    
    response = openai.Completion.create(engine="davinci", prompt=pass_prompt, temperature=temperature, max_tokens=330, presence_penalty=0.05, stop=['\n\n','\n\n\n',PROMPT_TAG])
    
    reply = response["choices"][0]["text"]
    reply=reply.strip('\n')
    reply=reply.strip()
    print(reply)

    print("REPLY:"+ reply)

    story_prompt_file.close()
    story_prompt_file= open(story_prompt_file_path, "a") 
    stored_story_file= open(stored_story_file_path,'a')

    story_prompt_file.write(constructed_prompt+ reply)
    story_prompt_file.close()

    reply_clean= reply.strip("[[Story]]:")
    stored_story_file.write(reply_clean)
    stored_story_file.close()
    
    return reply_clean

#User chooses to let the story continue on its own by GPT-3
def continue_story_without_caption(temperature = 0.8):

    story_prompt_file= open(story_prompt_file_path, "r") 
    story_so_far= story_prompt_file.read().strip('\n')
    story_so_far=story_so_far.strip()


    pass_prompt = orig_prompt +"\n\n\n" + story_so_far 

    print(f"the pass prompt is {pass_prompt}")
    
    response = openai.Completion.create(engine="davinci", prompt=pass_prompt,temperature=temperature, presence_penalty=0.05, max_tokens=400, stop=['\n\n','\n\n\n',PROMPT_TAG])
    
    reply = response["choices"][0]["text"]  
    print(reply)
    reply=reply.strip('\n')
    reply=reply.strip()

    story_prompt_file.close()
    story_prompt_file= open(story_prompt_file_path, "a") 
    stored_story_file= open(stored_story_file_path,'a')

    story_prompt_file.write(reply)
    story_prompt_file.close()

    reply_clean= reply.strip("[[Story]]:")
    stored_story_file.write(reply_clean)
    stored_story_file.close()
    
    return reply_clean 
     	


#User chooses to pass some text as input for continuing the story. 
def continue_story_with_text(user_text, temperature = 0.55):
    return continue_story_with_caption(user_text)


caption= "A man on a motorcycle"


new_story_with_caption(caption)

#continue_story_with_caption(caption)

#continue_story_without_caption()



