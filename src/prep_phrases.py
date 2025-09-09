import argparse

parser = argparse.ArgumentParser()

parser.add_argument("task", help="the task determines whether you generate coming out (comeout) or break up (breakup) prompts",type=str)

args = parser.parse_args()

partner_list = [r"[a-z]+friend",r"wife",r"spouse",r"husband",r"fianc(e|é)+",r"partner",r"girl",r"man",r"guy",r"him",r"her",r"them",r"someone",r"person",r"bf",r"gf"]

partner_templates = [r"break(ing)?( |-)?up with? (my |a )?({target})",r"end(ing)? (it|things) with? (my |a )?({target})",r"finish(ing)? (things )?with? (my |a )?({target})",r"call(ing)? (it|things) off with? (my |a )?({target})",r"tell(ing)? (my |a )?({target}) (that )*(it('| i)?s|we('| a)?re|things are) (finished|over|through|done|thru)",r"tell(ing)? (my |a )?({target}) (that )*I('| a)*m (going to |planning on |gonna |finna )*leav[a-z]+",r"leav(e|ing)? (my |a )?({target})$",r"(I|we('?ve)?) (need to|should|have to|got to|gotta|gonna)? break( |-)?up"]

identity_list = [r"trans[a-z]*", r"(?!het)[a-z]+sexual",r"(?!cis)[a-z]+gender", r"genderfluid", r"gay", r"(a )?lesbian", r"LGBT[A-Z]*", r"pansexual", r"queer", r"same gender loving", r"a bear", r"a butch", r"a cub", r"a dyke", r"a femme", r"a stud", r"a twink", r"latinx", r"non( |-)?binary", r"enby", r"transfem[a-z]*", r"transmasc(uline)?", r"two( |-)?spirit",r"queer",r"bi",r"pan",r"homo",r"ace"]

identity_templates = [
    r"com(e|ing)? out ([a-z-]+ )*(as ({target}))\b", 
    r"tell(ing)? ([a-z-]+ )?I('| a)?m ({target})\b", 
    r"let(ting)? ([a-z]+ )+know (that )?I('| a)?m ({target})\b",
    r"I('| a)?m ({target})\b[\s\S]*(com(e|ing)? out to|tell(ing)? my|let(ting)? ([a-z-]+ )+know)"]

#identity_templates = [r"^(?=.*com(e|ing)? out)(?=.*as ({target})\b).*",    r"^(?=.*tell(ing)? ([a-z-]+ )?)(?=.*I('| a)?m ({target})\b).*",    r"^(?=.*let(ing)? ([a-z-]+ )? know)(?=.*I('| a)?m ({target})\b).*"]

def create_prompts(prompt_list,phrases,templates):
    targets = "|".join(phrases)
    for template in templates:
        prompt_list.append(template.format(target=targets))
    return prompt_list

prompt_list = []

if args.task == "breakup":
    prompt_list = [r"call (off )*(my )*(wedding|engagement)( off )*",r"\bend (my )*(relationship|engagement)"]
    prompt_list = create_prompts(prompt_list,partner_list,partner_templates)
elif args.task == "comeout": 
    prompt_list = [r"com(e|ing)? out to (a |my )?[a-z]+[\?\.\!$]+"]
    prompt_list = create_prompts(prompt_list,identity_list,identity_templates)


with open(f"/home/eddie/wildchat/{args.task}prompts.txt","w",newline="") as f:
    for prompt in prompt_list:
        f.write(prompt+"\n")
