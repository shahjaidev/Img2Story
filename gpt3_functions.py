
#REQUIREMENT: pip install openai
import openai
import os
from tools import get_story_prompt_current_file,get_saved_story_new_file,get_new_story_prompt_file,create_file

openai.api_key='sk-XHaEg50XHxqsa7j3aSUx7YQNGsHTgmRWjNmRXh3k'
prompt_file_path= './GPT3Prompts/origin_prompt.txt'
story_prompt_file_path= './GPT3Prompts/story_so_far.txt'
#get_story_prompt_current_file()
#print("story_prompt_file_path", story_prompt_file_path)
#'./GPT3Prompts/story_so_far.txt'
display_story_file_path= './SavedStories/saved_story.txt'
#get_saved_story_new_file()



def new_story_with_caption(caption, temperature = 0.84):
    #create_file(story_prompt_file_path,display_story_file_path)
    prompt_file=open(prompt_file_path,'r')
    story_prompt_file=open(story_prompt_file_path, "w+")
    display_story_file=open(display_story_file_path, "w+")
    file_path = story_prompt_file_path
    # filesize = os.stat(file_path).st_size
#     if filesize !=0:
#         story_prompt_new_file_path= get_new_story_prompt_file(story_prompt_file_path)
#         story_prompt_file= open(story_prompt_new_file_path, "a+")
#         file_path = story_prompt_new_file_path

    original_prompt = prompt_file.read()
    constructed_prompt= "START: "+ caption
    #only when we fed in a new image
    pass_prompt = original_prompt +"\n\n\n" + constructed_prompt

    response = openai.Completion.create(engine="davinci", prompt=pass_prompt, max_tokens=150, stop=['\n\n'])

    reply = response["choices"][0]["text"]
    print(reply)

    story_prompt_file.write(constructed_prompt+ "\n")

    story_prompt_file.write(reply)

    display_story_file.write(reply)
    story_prompt_file.close()
    display_story_file.close()

    return reply

#When user chooses to continue the story through uploading an image
def continue_story_with_caption(caption, temperature = 0.84):
    #create_file(story_prompt_file_path,display_story_file_path)
    prompt_file=open(prompt_file_path,'r')
    story_prompt_file=open(story_prompt_file_path, "a+")
    display_story_file=open(display_story_file_path, "a+")

    original_prompt = prompt_file.read()
    story_so_far= story_prompt_file.read()

    constructed_prompt= "\n\n"+"ANCHOR: "+ caption

    original_prompt = prompt_file.read()
    constructed_prompt= "START: "+ caption
    #only when we fed in a new image
    pass_prompt = original_prompt +"\n\n\n" + story_so_far+ constructed_prompt

    response = openai.Completion.create(engine="davinci", prompt=pass_prompt, max_tokens=150, stop=['\n\n'])

    reply = response["choices"][0]["text"]
    print(reply)

    story_prompt_file.write(constructed_prompt+ "\n")

    story_prompt_file.write(reply)

    display_story_file.write(reply)
    display_story_file.write(reply)
    story_prompt_file.close()
    display_story_file.close()

    return reply

#User chooses to let the story continue on its own by GPT-3
def continue_story_without_caption(temperature = 0.84):
    #create_file(story_prompt_file_path,display_story_file_path)
    prompt_file=open(prompt_file_path,'r')
    story_prompt_file=open(story_prompt_file_path, "a+")
    display_story_file=open(display_story_file_path, "a+")
    original_prompt = prompt_file.read()
    story_so_far= story_prompt_file.read()

    original_prompt = prompt_file.read()
    constructed_prompt= "\n\n"+ "CONTINUE: "

    #Continue option is chosen by user
    pass_prompt = original_prompt +"\n\n\n" + story_so_far+ constructed_prompt

    response = openai.Completion.create(engine="davinci", prompt=pass_prompt, max_tokens=150, stop=['\n\n'])

    reply = response["choices"][0]["text"]
    print(reply)

    story_prompt_file.write(constructed_prompt+ "\n")

    story_prompt_file.write(reply)

    display_story_file.write(reply)
    story_prompt_file.close()
    display_story_file.close()

    return reply


#User chooses to pass some text as input for continuing the story.
def continue_story_with_text(user_text, temperature = 0.84):
    #create_file(story_prompt_file_path,display_story_file_path)
    prompt_file=open(prompt_file_path,'r')
    story_prompt_file=open(story_prompt_file_path, "a+")
    display_story_file=open(display_story_file_path, "a+")
    original_prompt = prompt_file.read()
    story_so_far= story_prompt_file.read()

    constructed_prompt= "\n\n"+"ANCHOR: "+ user_text

    original_prompt = prompt_file.read()
    constructed_prompt= "START: "+ caption
    #only when we fed in a new image
    pass_prompt = original_prompt +"\n\n\n" + story_so_far+ constructed_prompt

    response = openai.Completion.create(engine="davinci", prompt=pass_prompt, max_tokens=150, stop=['\n\n'])

    reply = response["choices"][0]["text"]
    print(reply)

    story_prompt_file.write(constructed_prompt+ "\n")

    story_prompt_file.write(reply)

    display_story_file.write(reply)
    story_prompt_file.close()
    display_story_file.close()

    return reply
