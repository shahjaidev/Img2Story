import os
def get_story_prompt_current_file():
	story_prompt_folder= './GPT3Prompts/'
	path, dirs, files = next(os.walk(story_prompt_folder))
	files = [ fi for fi in files if fi.endswith(".txt") ]
	file_count = len(files)
	#file_version=-1
	#filename ="story_so_far.txt"
	if file_count==1:
		file_version = file_count
	else:
		file_version = file_count-1

	#one filr is original prom

	file_path = story_prompt_folder + "story_so_far_"+str(file_version)+".txt"
	story_prompt_file=open(file_path, "a+")
	story_prompt_file.close()
	#if not os.path.exists(file_path):
	#	os.makedirs(file_path)
	return file_path

def get_saved_story_new_file():
	saved_story_folder= './SavedStories/'
	path, dirs, files = next(os.walk(saved_story_folder))
	files = [ fi for fi in files if fi.endswith(".txt") ]
	file_count = len(files)
	#file_version=-1
	#filename ="story_so_far.txt"

	file_version = file_count+1

	file_path = saved_story_folder + "saved_story_"+str(file_version)+".txt"
	#if not os.path.exists(file_path):
	#	os.makedirs(file_path)
	return file_path

def get_new_story_prompt_file(prompt_file_path):
	filename =os.path.basename(prompt_file_path)
	file_name, file_extension = os.path.splitext(filename)
	version = file_name[-1]
	file_version = int(version)+1
	file_path='./GPT3Prompts/' + "story_so_far_"+str(file_version)+".txt"
	#if not os.path.exists(file_path):
	#	os.makedirs(file_path)
	return file_path

def create_file(story_prompt_file_path,display_story_file_path):
	#prompt_file=open(prompt_file_path,'r')
	story_prompt_file=open(story_prompt_file_path, "a+")
	display_story_file=open(display_story_file_path, "a+")


	story_prompt_file.close()
	display_story_file.close()
	return






