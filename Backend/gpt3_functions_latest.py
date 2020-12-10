
#!/usr/bin/env python

#REQUIREMENT: pip install openai

import openai


#orig_prompt= "Here is an award winning short story:"+ "\n"+ "A woman paddling on a surfboard. Maria sat on her surfboard, as it bobbed up and down on the coastal waves. The neon lights of Ocean Drive burned on the shore in the distance, like green and pink embers. It was just her, the water, and whatever starlight got past the Miami light pollution. A shark swimming in the ocean. She was paddling back to shore when she saw a shark fin in the water. At first she thought it was a manatee or a dolphin. She’d occasionally seen those swimming around here, spent countless hours watching them under the night sky. But this was different. A boat in the lake. A bright spotlight shown down on Maria. Maria looked up, and saw a small fishing boat, floating half-way between her and the shoreline. A stout man in an orange rain coat and fishing boots stood on the deck. “Get out of the water!” A gruff voice shouted. Maria's eyes went wide. She turned and saw the large dorsal fin rapidly advancing. The shark’s body hovered just under the surface like a dark cloud. Maria’s hands dove into the water, paddling faster than she’d ever paddled in her life. Her legs kicked frantically. "

prompt_file_path= './GPT3Prompts/origin_prompt.txt'

prompt_file= open(prompt_file_path,'r')

orig_prompt= prompt_file.read()

#print(orig_prompt)


story_prompt_file_path= './GPT3Prompts/story_so_far.txt'

openai.api_key = "sk-XHaEg50XHxqsa7j3aSUx7YQNGsHTgmRWjNmRXh3k" 




def new_story_with_caption(caption, temperature = 0.73):

    story_prompt_file= open(story_prompt_file_path, "w+") 

    constructed_prompt=  "Here is an award winning short story:"+ "\n"+ caption
    pass_prompt = orig_prompt +"\n\n\n" + constructed_prompt 
    
    response = openai.Completion.create(engine="davinci", prompt=pass_prompt, max_tokens=150, stop=['\n','\n\n','\n\n\n'])
    
    reply = response["choices"][0]["text"]  
    print(reply)
    
    constructed_prompt=  "Here is an award winning short story:"+ "\n"
    story_prompt_file.write(constructed_prompt)


    story_prompt_file.write(reply)


    story_prompt_file.close()
    
    return reply 	

#When user chooses to continue the story through uploading an image
def continue_story_with_caption(caption, temperature = 0.73):
 
    story_prompt_file= open(story_prompt_file_path, "r") 
    story_so_far= story_prompt_file.read()

    constructed_prompt=  story_so_far+ " "+ caption + "."
    pass_prompt = orig_prompt +"\n\n\n" + constructed_prompt 
    
    response = openai.Completion.create(engine="davinci", prompt=pass_prompt, max_tokens=150, stop=['\n','\n\n','\n\n\n'])
    
    reply = response["choices"][0]["text"]  
    print(reply)

    story_prompt_file.close()
    story_prompt_file= open(story_prompt_file_path, "a") 

    story_prompt_file.write(reply)
    story_prompt_file.close()
    
    return reply 

#User chooses to let the story continue on its own by GPT-3
def continue_story_without_caption(temperature = 0.73):

    story_prompt_file= open(story_prompt_file_path, "r") 
    story_so_far= story_prompt_file.read()


    pass_prompt = orig_prompt +"\n\n\n" + story_so_far 
    
    response = openai.Completion.create(engine="davinci", prompt=pass_prompt, max_tokens=150, stop=['\n','\n\n','\n\n\n'])
    
    reply = response["choices"][0]["text"]  
    print(reply)

    story_prompt_file.close()
    story_prompt_file= open(story_prompt_file_path, "a") 

    story_prompt_file.write(reply)
    story_prompt_file.close()
    
    return reply 
     	


#User chooses to pass some text as input for continuing the story. 
def continue_story_with_text(user_text, temperature = 0.73):
    return continue_story_with_caption(user_text)


caption= "a large plane."
#continue_story_without_caption()


#continue_story_with_caption(caption)

new_story_with_caption(caption)

#new_story_with_caption(caption)


