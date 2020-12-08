
#REQUIREMENT: pip install openai

import openai

prompt_file_path= './GPT3Prompts/origin_prompt.txt'
story_file_path= './GPT3Prompts/story_so_far.txt'
display_story_file_path= './SavedStory/saved_story.txt'

openai.api_key = "org-KCAwDUnNQR2Qw3mqomhiDoXF" 




prompt_file= open(prompt_file_path,'r') 
story_prompt_file= open(story_prompt_file_path, ‘rw’) 
display_story_file=open(display_story_file_path, ‘rw’) 


def new_story_with_caption(caption, training=danabot, temperature = 0.84):

    original_prompt = prompt_file.read()
    constructed_prompt= "START: "+ caption
	pass_prompt = original_prompt +"\n\n\n" + constructed_prompt // only when we fed in a new image
	response = openai.Completion.create(engine="davinci", prompt=pass_prompt, max_tokens=150, stop=['\n\n'])

	reply = response["choices"][0]["text"]  
    print(reply)

	story_file.write(constructed_prompt+ ‘\n’)
	story_file.write(reply)
	return reply 	

#When user chooses to continue the story through uploading an image
def response_with_caption(caption, temperature = 0.84):

    original_prompt = prompt_file.read()
    story_so_far= story_file.read()

    constructed_prompt= "\n\n"+"ANCHOR: "+ caption
	pass_prompt = original_prompt +"\n\n" + story_so_far + constructed_prompt 
	response = openai.Completion.create(engine="davinci", prompt=pass_prompt, max_tokens=150, stop=['\n\n'])

	reply = response["choices"][0]["text"]  
    print(reply)
    
	story_file.write(constructed_prompt+ ‘\n’)
	story_file.write(reply)
	return reply 

#User chooses to let the story continue on its own by GPT-3
def continue_story_without_caption(temperature = 0.84):
    prompt = prompt_file.read()
    story_so_far= story_file.read()
    constructed_prompt= "Anchor: "+ caption
	pass_prompt = prompt+ story_so_far+constructed_prompt // only when we fed in a new image
	response = openai.Completion.create(engine="davinci", prompt=pass_prompt, max_tokens=130, stop=['\n\n'])

	reply = response["choices"][0]["text"]  

	story_file.write(constructed_prompt+ ‘\n’)
	story_file.write(reply)
	return reply 		


#User chooses to pass some text as input for continuing the story. 
def continue_story_with_text(user_text, temperature = 0.84):
    prompt = prompt_file.read()
    story_so_far= story_file.read()
    constructed_prompt= "Anchor: "+ user_text
	pass_prompt = prompt+ story_so_far+constructed_prompt // only when we fed in a new image
	response = openai.Completion.create(engine="davinci", prompt=pass_prompt, max_tokens=130, stop=['\n\n'])

	reply = response["choices"][0]["text"]  

	story_file.write(constructed_prompt+ ‘\n’)
	story_file.write(reply)
	return reply 	



caption= "A man standing in the sand with a dog"

