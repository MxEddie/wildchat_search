import argparse

parser = argparse.ArgumentParser()

parser.add_argument("task", help="the task determines whether you generate coming out (comeout) or break up (breakup) prompts",type=str)

args = parser.parse_args()

partner_list = [r"[a-z]?friend",r"wife",r"spouse",r"husband",r"fianc(e|é)+",r"partner",r"girl",r"man"]

partner_templates = [r"break up with my {partner}",r"end (it|things) with my {partner}",r"finish (things )* with my {partner}",r"call (it|things) off with my",r"tell my {partner} (that )*(it'?s|we'?re|things are) (finished|over|through|done)",r"tell my {partner} (that )*I('| a)*m leaving",r"leave my {partner}"]


identity_list = [r"trans[a-z]*", r"[a-z]?sexual", r"genderfluid", r"gay", r"(a )*lesbian", r"LGBT[A-Z]*", r"pansexual", r"queer", r"same gender loving", r"a bear", r"a butch", r"a cub", r"a dyke", r"a femme", r"a stud", r"a twink", r"latinx", r"non-*binary", r"transfem[a-z]*", r"transmasc(uline)*", r"two-*spirit"]

identity_templates = [r"come out (to )*(my )*([a-z] )*as {identity}", r"tell (my )*([a-z] )*(that )*I('| a)*m {identity}"]

def create_prompts(prompt_list,phrases,templates):
    for template in templates:
        for phrase in phrases:
            prompt_list.append(template.format(identity=phrase))
    return prompt_list

prompt_list = []

if args.task == "breakup":
    prompt_list = [r"call off my (wedding|engagement)",r"end my relationship"]
    prompt_list = create_prompts(prompt_list,partner_list,partner_templates)
elif args.task == "comeout": 
    prompt_list = create_prompts(prompt_list,identity_list,identity_templates)


with open(f"/home/eddie/wildchat/{args.task}prompts.txt","w",newline="") as f:
    for prompt in prompt_list:
        f.write(prompt+"\n")
