"""
File containing regex templates for searching conversations in the Wildchat and LMSys-Chat datasets. The templates are organized into categories such as advice, roleplay, breakup, queer, and comeout. Each category contains a list of regex patterns that can be used to identify relevant conversations.
"""
TEST_TEMPLATES = {
    "first_list": [
        r"(help|advise) me"
    ],
    "second_list": [
        r"cook[a-z]*", r"bake[a-z]*"
    ]
}

# Define keywords which will be searched for 
AP4L_TEMPLATES = {
    "advice": [
        r"(help|advise) me",
        r"(what |how )?(can|do|should|could) I",
        r"what'?s the [a-z]+st way to",
        r"is it [a-z]+ to",
        r"(tell|teach|show) me how (to|I)",
        r"(I('?d)* (really )*(want|need|(would )?like)|give me) (your |some |any )?(advice|help|suggestions?|thoughts?|input|ideas?)",
        r"how to do (it|that)",
        r"is (this|that|it) an? [a-z]+ idea",
        r"is my idea [a-z]+",
        r"do you have (any |some )?(advice|help|suggestions?|thoughts?|input|ideas?)",
        r"what do you think (about|of) (it|that|my|this)"
    ], 
    "roleplay": [   
        r"pretend (to be|you('| a)?re?) (my|him|her|them|someone|an?)",
        r"(role( |-)?)?play as (my|him|her|them|someone|an?)",
        r"(pretend|respond|talk|act|behave|speak) (to me )*(like|as( if)?) (you( we| a|')?re? )?(my|him|her|them|someone|an?)", 
        r"practi[cs]e [a-z]*ing (to )*(my|him|her|them|someone|an?)"
    ],
    "breakup":  [
        r"break(ing)?( |-)?up with?",r"end(ing)? (it|things) with?",r"finish(ing)? (things )?with?",r"call(ing)? (it|things) off with?",r"tell(ing)? (my |a )? ([a-z]+ )+ (that )*(it('| i)?s|we('| a)?re|things are) (finished|over|through|done|thru)",r"tell(ing)? ([a-z]+ )+ (that )*I('| a)*m (going to |planning on |gonna |finna )*leav[a-z]+",r"leav(e|ing)? (my |a ) [a-z]+\W?$",r"(I|we('?ve)?) (need to|should|have to|got to|gotta|gonna)? break( |-)?up", r"call (off )*(my )*(wedding|engagement)( off )*",r"\bend (my )*(relationship|engagement)"
    ],
    "queer": [
        r"trans[a-z]*", r"(?!het)[a-z]+sexual",r"(?!cis)[a-z]+gender", r"genderfluid", r"gay", r"(a )?lesbian", r"LGBT[A-Z]*", r"pansexual", r"queer", r"same gender loving", r"a bear", r"a butch", r"a cub", r"a dyke", r"a femme", r"a stud", r"a twink", r"latinx", r"non( |-)?binary", r"enby", r"transfem[a-z]*", r"transmasc(uline)?", r"two( |-)?spirit",r"queer",r"bi",r"pan",r"homo",r"ace"],
    "comeout": [
        r"com(e|ing)? out", r"tell(ing)? ([a-z-]+ )?I('| a)?m", r"let(ting)? ([a-z]+ )+know (that )?I('| a)?m"
    ]
}

AIPV_TEMPLATES = {
    "IPV": [
        r"IPV",r"abuse",r"violen(t|ce)",r"toxic",r"control[a-z]*",r"manipulat[a-z]*",r"abuse",r"isolat[a-z]+",r"threat[a-z]*",r"intimidat[a-z]+",r"stalk[a-z]*",r"harass[a-z]*",r"jealous[a-z]*",r"blam[a-z]*",r"put[a-z]* (me|you) down",r"mak[a-z]* (me|you) feel (stupid|worthless|scared)",r"check[a-z]* (my|your) (phone|emails|texts|messages|facebook|insta(gram)|IG|snap(chat)?)",r"accus[a-z]* (me|you) of cheating",r"force[a-z]* (me|you) to"],
    "relationship": [
            r"relationship",r"[a-z]friend",r"bf",r"gf",r"partner",r"wife",r"husband",r"my SO\b",r"dating",r"marriage",r"married",r"fiance(e|é)+",r"engaged",r"spouse",r"significant other",r"my (wo)?man",r"my girl",r"hubby",r"my ex\b"
        ], 
    }