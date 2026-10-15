"""Build data/luganda_eval_items.xlsx: the draft Luganda evaluation items for native-speaker validation.

All items are drafts. A native speaker sets STATUS for each row (OK / FIXED / DROP) and, for FIXED rows,
writes the corrected text in the fixed_* columns. The evaluation notebooks use only OK and FIXED rows.
"""
import sys

from openpyxl import Workbook
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

# (id suffix, correct, wrong, english, changed morpheme, note, my confidence)
NOUN_CLASS = [
    # adjective concord
    ("Ekitabo ekinene kiri ku mmeeza.", "Ekitabo ennene kiri ku mmeeza.", "The big book is on the table.", "eki- → en- (cl7 → cl9 adjective)", "", "high"),
    ("Ebitabo ebinene biri ku mmeeza.", "Ebitabo amanene biri ku mmeeza.", "The big books are on the table.", "ebi- → ama- (cl8 → cl6)", "", "high"),
    ("Omusajja omunene ali mu nnyumba.", "Omusajja ekinene ali mu nnyumba.", "The big man is in the house.", "omu- → eki- (cl1 → cl7)", "", "high"),
    ("Abasajja abanene bali mu nnyumba.", "Abasajja omunene bali mu nnyumba.", "The big men are in the house.", "aba- → omu- (cl2 → cl1)", "", "high"),
    ("Omuti omunene guli mu nnimiro.", "Omuti ekinene guli mu nnimiro.", "The big tree is in the garden.", "omu- → eki- (cl3 → cl7)", "", "high"),
    ("Emiti eminene giri mu nnimiro.", "Emiti ebinene giri mu nnimiro.", "The big trees are in the garden.", "emi- → ebi- (cl4 → cl8)", "", "high"),
    ("Ennyumba ennene eri ku lusozi.", "Ennyumba ekinene eri ku lusozi.", "The big house is on the hill.", "en- → eki- (cl9 → cl7)", "", "high"),
    ("Ejjinja eddene liri ku kkubo.", "Ejjinja ennene liri ku kkubo.", "The big stone is on the road.", "eri- (eddene) → en- (cl5 → cl9)", "Check the cl5 form eddene.", "medium"),
    ("Amayinja amanene gali ku kkubo.", "Amayinja ebinene gali ku kkubo.", "The big stones are on the road.", "ama- → ebi- (cl6 → cl8)", "", "high"),
    ("Olutindo olunene luli ku mugga.", "Olutindo ekinene luli ku mugga.", "The big bridge is on the river.", "olu- → eki- (cl11 → cl7)", "", "high"),
    ("Akambe akatono kali ku mmeeza.", "Akambe ekitono kali ku mmeeza.", "The small knife is on the table.", "aka- → eki- (cl12 → cl7)", "", "high"),
    ("Omwana omulungi ali ku ssomero.", "Omwana ekirungi ali ku ssomero.", "The good child is at school.", "omu- → eki- (cl1 → cl7)", "", "high"),
    ("Abaana abalungi bali ku ssomero.", "Abaana ebirungi bali ku ssomero.", "The good children are at school.", "aba- → ebi- (cl2 → cl8)", "", "high"),
    ("Ekikopo ekitono kiri mu ffumbiro.", "Ekikopo akatono kiri mu ffumbiro.", "The small cup is in the kitchen.", "eki- → aka- (cl7 → cl12)", "", "high"),
    ("Embwa ennene eri mu luggya.", "Embwa omunene eri mu luggya.", "The big dog is in the courtyard.", "en- → omu- (cl9 → cl1)", "", "high"),
    ("Emmotoka empya eri mu luggya.", "Emmotoka ekiggya eri mu luggya.", "The new car is in the courtyard.", "em- (empya) → eki- (cl9 → cl7)", "Check -ggya forms: empya / ekiggya.", "medium"),
    ("Ekitabo ekiggya kiri mu nsawo.", "Ekitabo empya kiri mu nsawo.", "The new book is in the bag.", "eki- → em- (cl7 → cl9)", "Check -ggya forms.", "medium"),
    ("Omukazi omuwanvu ali mu katale.", "Omukazi abawanvu ali mu katale.", "The tall woman is in the market.", "omu- → aba- (cl1 → cl2)", "", "high"),
    ("Abakazi abawanvu bali mu katale.", "Abakazi omuwanvu bali mu katale.", "The tall women are in the market.", "aba- → omu- (cl2 → cl1)", "", "high"),
    ("Omugga omuwanvu guli mu kyalo.", "Omugga ekiwanvu guli mu kyalo.", "The long river is in the village.", "omu- → eki- (cl3 → cl7)", "", "high"),
    # possessive concord
    ("Ekitabo kyange kiri wano.", "Ekitabo wange kiri wano.", "My book is here.", "kya- → wa- (cl7 → cl1 possessive)", "", "high"),
    ("Ebitabo byange biri wano.", "Ebitabo kyange biri wano.", "My books are here.", "bya- → kya- (cl8 → cl7)", "", "high"),
    ("Omwana wange ali wano.", "Omwana kyange ali wano.", "My child is here.", "wa- → kya- (cl1 → cl7)", "", "high"),
    ("Abaana bange bali wano.", "Abaana wange bali wano.", "My children are here.", "ba- → wa- (cl2 → cl1)", "", "high"),
    ("Embwa yange eri wano.", "Embwa kyange eri wano.", "My dog is here.", "ya- → kya- (cl9 → cl7)", "", "high"),
    ("Embwa zange ziri wano.", "Embwa yange ziri wano.", "My dogs are here.", "za- → ya- (cl10 → cl9)", "embwa is cl9 and cl10; the verb ziri shows that it is plural.", "medium"),
    ("Omuti gwange guli wano.", "Omuti kyange guli wano.", "My tree is here.", "gwa- → kya- (cl3 → cl7)", "", "high"),
    ("Emiti gyange giri wano.", "Emiti byange giri wano.", "My trees are here.", "gya- → bya- (cl4 → cl8)", "", "high"),
    ("Olugoye lwange luli wano.", "Olugoye kyange luli wano.", "My cloth is here.", "lwa- → kya- (cl11 → cl7)", "", "high"),
    ("Amagi gange gali wano.", "Amagi byange gali wano.", "My eggs are here.", "ga- → bya- (cl6 → cl8)", "", "high"),
    ("Eggi lyange liri wano.", "Eggi gange liri wano.", "My egg is here.", "lya- → ga- (cl5 → cl6)", "", "high"),
    ("Akambe kange kali wano.", "Akambe kyange kali wano.", "My knife is here.", "ka- → kya- (cl12 → cl7)", "", "high"),
    ("Ennyumba yange eri wano.", "Ennyumba lyange eri wano.", "My house is here.", "ya- → lya- (cl9 → cl5)", "", "high"),
    ("Emmotoka yaffe eri wano.", "Emmotoka kyaffe eri wano.", "Our car is here.", "ya- → kya- (cl9 → cl7)", "", "high"),
    ("Ekibuga kyaffe kinene.", "Ekibuga waffe kinene.", "Our town is big.", "kya- → wa- (cl7 → cl1)", "", "high"),
    # demonstrative concord
    ("Omuti guno muwanvu.", "Omuti kino muwanvu.", "This tree is tall.", "gu- → ki- (cl3 → cl7 demonstrative)", "", "high"),
    ("Emiti gino miwanvu.", "Emiti bino miwanvu.", "These trees are tall.", "gi- → bi- (cl4 → cl8)", "", "high"),
    ("Ekitabo kino kirungi.", "Ekitabo guno kirungi.", "This book is good.", "ki- → gu- (cl7 → cl3)", "", "high"),
    ("Ebitabo bino birungi.", "Ebitabo gino birungi.", "These books are good.", "bi- → gi- (cl8 → cl4)", "", "high"),
    ("Omwana ono mulwadde.", "Omwana kino mulwadde.", "This child is sick.", "o- (ono) → ki- (cl1 → cl7)", "", "high"),
    ("Abaana bano balwadde.", "Abaana ono balwadde.", "These children are sick.", "ba- → o- (cl2 → cl1)", "", "high"),
    ("Ennyumba eno nnene.", "Ennyumba kino nnene.", "This house is big.", "e- (eno) → ki- (cl9 → cl7)", "", "high"),
    ("Ennyumba zino nnene.", "Ennyumba bino nnene.", "These houses are big.", "zi- → bi- (cl10 → cl8)", "", "high"),
    ("Amagi gano malungi.", "Amagi kino malungi.", "These eggs are good.", "ga- → ki- (cl6 → cl7)", "", "high"),
    ("Olugoye luno lulungi.", "Olugoye kino lulungi.", "This cloth is good.", "lu- → ki- (cl11 → cl7)", "", "high"),
    ("Akambe kano katono.", "Akambe kino katono.", "This knife is small.", "ka- → ki- (cl12 → cl7)", "", "high"),
    ("Ejjinja lino ddene.", "Ejjinja kino ddene.", "This stone is big.", "li- → ki- (cl5 → cl7)", "Check the predicate form ddene.", "medium"),
    ("Omugga guno muwanvu.", "Omugga ono muwanvu.", "This river is long.", "gu- → o- (cl3 → cl1)", "", "high"),
    ("Ekibuga kino kinene.", "Ekibuga eno kinene.", "This town is big.", "ki- → e- (cl7 → cl9)", "", "high"),
    ("Embwa zino nnungi.", "Embwa bano nnungi.", "These dogs are good.", "zi- → ba- (cl10 → cl2)", "", "high"),
]

SUBJ_OBJ = [
    # subject marker agrees with the noun
    ("Omwana agudde.", "Omwana bagudde.", "The child has fallen.", "a- → ba- (SM cl1 → cl2)", "", "high"),
    ("Abaana bagudde.", "Abaana agudde.", "The children have fallen.", "ba- → a- (SM cl2 → cl1)", "", "high"),
    ("Omuti gugudde.", "Omuti kigudde.", "The tree has fallen.", "gu- → ki- (SM cl3 → cl7)", "", "high"),
    ("Emiti gigudde.", "Emiti bigudde.", "The trees have fallen.", "gi- → bi- (SM cl4 → cl8)", "", "high"),
    ("Ekitabo kigudde.", "Ekitabo gugudde.", "The book has fallen.", "ki- → gu- (SM cl7 → cl3)", "", "high"),
    ("Ebitabo bigudde.", "Ebitabo gigudde.", "The books have fallen.", "bi- → gi- (SM cl8 → cl4)", "", "high"),
    ("Ejjinja ligudde.", "Ejjinja kigudde.", "The stone has fallen.", "li- → ki- (SM cl5 → cl7)", "", "medium"),
    ("Amayinja gagudde.", "Amayinja bigudde.", "The stones have fallen.", "ga- → bi- (SM cl6 → cl8)", "", "high"),
    ("Oluggi lugudde.", "Oluggi kigudde.", "The door has fallen.", "lu- → ki- (SM cl11 → cl7)", "", "high"),
    ("Akaana kakaaba.", "Akaana kikaaba.", "The little child is crying.", "ka- → ki- (SM cl12 → cl7)", "", "high"),
    ("Enkuba etonnya.", "Enkuba atonnya.", "The rain is falling.", "e- → a- (SM cl9 → cl1)", "", "high"),
    ("Omuliro gwaka.", "Omuliro kyaka.", "The fire is burning.", "gw- → ky- (SM cl3 → cl7)", "", "high"),
    ("Omusana gwaka.", "Omusana bwaka.", "The sun is shining.", "gw- → bw- (SM cl3 → cl14)", "", "medium"),
    ("Ebinyonyi biyimba.", "Ebinyonyi giyimba.", "The birds are singing.", "bi- → gi- (SM cl8 → cl4)", "", "high"),
    ("Ekinyonyi kibuuka.", "Ekinyonyi abuuka.", "The bird is flying.", "ki- → a- (SM cl7 → cl1)", "", "high"),
    ("Amazzi gakulukuta.", "Amazzi kikulukuta.", "The water is flowing.", "ga- → ki- (SM cl6 → cl7)", "Check the verb -kulukuta.", "medium"),
    ("Ebikoola bigwa.", "Ebikoola gigwa.", "The leaves are falling.", "bi- → gi- (SM cl8 → cl4)", "", "high"),
    ("Abagenyi batuuse.", "Abagenyi atuuse.", "The guests have arrived.", "ba- → a- (SM cl2 → cl1)", "", "high"),
    ("Abalimi balima kasooli.", "Abalimi alima kasooli.", "The farmers grow maize.", "ba- → a- (SM cl2 → cl1)", "", "high"),
    ("Omuyizi asoma ekitabo.", "Omuyizi basoma ekitabo.", "The student is reading a book.", "a- → ba- (SM cl1 → cl2)", "Not a kinship noun, so the honorific plural does not apply.", "high"),
    ("Abazimbi bazimba ennyumba.", "Abazimbi azimba ennyumba.", "The builders are building a house.", "ba- → a- (SM cl2 → cl1)", "", "high"),
    ("Emiti gikula mangu.", "Emiti gukula mangu.", "The trees grow fast.", "gi- → gu- (SM cl4 → cl3)", "", "high"),
    ("Olugoye lukaze.", "Olugoye kikaze.", "The cloth has dried.", "lu- → ki- (SM cl11 → cl7)", "", "medium"),
    ("Ekimuli kifudde.", "Ekimuli gufudde.", "The flower has withered.", "ki- → gu- (SM cl7 → cl3)", "", "medium"),
    ("Embwa ebuze.", "Embwa kibuze.", "The dog is lost.", "e- → ki- (SM cl9 → cl7)", "", "high"),
    # subject marker agrees with the pronoun
    ("Nze ŋŋenda ku ssomero.", "Nze agenda ku ssomero.", "I am going to school.", "n- (ŋŋ) → a- (SM 1sg → 3sg)", "The spelling ngenda is in the variation sheet.", "high"),
    ("Ffe tugenda ku ssomero.", "Ffe bagenda ku ssomero.", "We are going to school.", "tu- → ba- (SM 1pl → 3pl)", "", "high"),
    ("Mmwe mugenda ku ssomero.", "Mmwe tugenda ku ssomero.", "You (pl.) are going to school.", "mu- → tu- (SM 2pl → 1pl)", "", "high"),
    ("Bo bagenda ku ssomero.", "Bo tugenda ku ssomero.", "They are going to school.", "ba- → tu- (SM 3pl → 1pl)", "", "high"),
    ("Ye agenda ku ssomero.", "Ye mugenda ku ssomero.", "He/she is going to school.", "a- → mu- (SM 3sg → 2pl)", "", "high"),
    ("Ggwe olya emmere.", "Ggwe alya emmere.", "You (sg.) are eating food.", "o- → a- (SM 2sg → 3sg)", "", "high"),
    ("Nze ndya emmere.", "Nze olya emmere.", "I am eating food.", "n- → o- (SM 1sg → 2sg)", "", "high"),
    ("Ffe tulya emmere.", "Ffe mulya emmere.", "We are eating food.", "tu- → mu- (SM 1pl → 2pl)", "", "high"),
    ("Bo balya emmere.", "Bo alya emmere.", "They are eating food.", "ba- → a- (SM 3pl → 3sg)", "", "high"),
    ("Mmwe mulya emmere.", "Mmwe balya emmere.", "You (pl.) are eating food.", "mu- → ba- (SM 2pl → 3pl)", "", "high"),
    # object marker agrees with the topic noun
    ("Ekitabo nkisomye.", "Ekitabo ngisomye.", "The book, I have read it.", "-ki- → -gi- (OM cl7 → cl9)", "", "high"),
    ("Ebitabo mbisomye.", "Ebitabo nkisomye.", "The books, I have read them.", "-bi- → -ki- (OM cl8 → cl7)", "", "high"),
    ("Embwa ngirabye.", "Embwa nkirabye.", "The dog, I have seen it.", "-gi- → -ki- (OM cl9 → cl7)", "", "high"),
    ("Omwana mmulabye.", "Omwana mbalabye.", "The child, I have seen him/her.", "-mu- → -ba- (OM cl1 → cl2)", "", "high"),
    ("Abaana mbalabye.", "Abaana mmulabye.", "The children, I have seen them.", "-ba- → -mu- (OM cl2 → cl1)", "", "high"),
    ("Emmere ngiridde.", "Emmere nkiridde.", "The food, I have eaten it.", "-gi- → -ki- (OM cl9 → cl7)", "", "high"),
    ("Amazzi nganywedde.", "Amazzi nginywedde.", "The water, I have drunk it.", "-ga- → -gi- (OM cl6 → cl9)", "", "medium"),
    ("Omuti ngutemye.", "Omuti nkitemye.", "The tree, I have cut it.", "-gu- → -ki- (OM cl3 → cl7)", "", "high"),
    ("Emiti ngitemye.", "Emiti ngutemye.", "The trees, I have cut them.", "-gi- → -gu- (OM cl4 → cl3)", "", "high"),
    ("Ekitabo nkiguze.", "Ekitabo mbiguze.", "The book, I have bought it.", "-ki- → -bi- (OM cl7 → cl8)", "", "high"),
    ("Amagi ngaguze.", "Amagi ngiguze.", "The eggs, I have bought them.", "-ga- → -gi- (OM cl6 → cl9)", "", "medium"),
    ("Akambe nkaguze.", "Akambe nkiguze.", "The knife, I have bought it.", "-ka- → -ki- (OM cl12 → cl7)", "", "high"),
    ("Olugoye nduguze.", "Olugoye nkiguze.", "The cloth, I have bought it.", "-lu- → -ki- (OM cl11 → cl7)", "Check n + lu → ndu.", "low"),
    ("Emmotoka ngiguze.", "Emmotoka nkiguze.", "The car, I have bought it.", "-gi- → -ki- (OM cl9 → cl7)", "", "high"),
    ("Abagenyi tubalabye.", "Abagenyi tumulabye.", "The guests, we have seen them.", "-ba- → -mu- (OM cl2 → cl1)", "", "high"),
]

TENSE = [
    # jjo (yesterday): past -a- is correct, future -li- is wrong
    ("Jjo nnagenda e Kampala.", "Jjo ndigenda e Kampala.", "Yesterday I went to Kampala.", "-a- → -li- (past → far future)", "", "high"),
    ("Jjo twagenda e Masaka.", "Jjo tuligenda e Masaka.", "Yesterday we went to Masaka.", "-a- → -li-", "", "high"),
    ("Jjo yagenda ku dduuka.", "Jjo aligenda ku dduuka.", "Yesterday he/she went to the shop.", "-a- → -li-", "", "high"),
    ("Jjo baagenda mu katale.", "Jjo baligenda mu katale.", "Yesterday they went to the market.", "-a- → -li-", "", "high"),
    ("Jjo nnalya matooke.", "Jjo ndirya matooke.", "Yesterday I ate matooke.", "-a- → -li-", "Check if matooke needs the augment (amatooke).", "medium"),
    ("Jjo twasoma ekitabo.", "Jjo tulisoma ekitabo.", "Yesterday we read a book.", "-a- → -li-", "", "high"),
    ("Jjo yalaba embwa.", "Jjo aliraba embwa.", "Yesterday he/she saw a dog.", "-a- → -li-", "", "high"),
    ("Jjo baalima ennimiro.", "Jjo balirima ennimiro.", "Yesterday they dug the garden.", "-a- → -li-", "", "high"),
    ("Jjo nnagula engoye.", "Jjo ndigula engoye.", "Yesterday I bought clothes.", "-a- → -li-", "", "high"),
    ("Jjo yakola nnyo.", "Jjo alikola nnyo.", "Yesterday he/she worked hard.", "-a- → -li-", "", "high"),
    ("Jjo baazina obulungi.", "Jjo balizina obulungi.", "Yesterday they danced well.", "-a- → -li-", "", "high"),
    ("Jjo twayimba mu kkanisa.", "Jjo tuliyimba mu kkanisa.", "Yesterday we sang in church.", "-a- → -li-", "", "high"),
    # enkya (tomorrow): near future -naa- is correct, past -a- is wrong
    ("Enkya nnaagenda e Kampala.", "Enkya nnagenda e Kampala.", "Tomorrow I will go to Kampala.", "-naa- → -a- (near future → past)", "", "high"),
    ("Enkya tunaagenda e Jinja.", "Enkya twagenda e Jinja.", "Tomorrow we will go to Jinja.", "-naa- → -a-", "", "high"),
    ("Enkya anaagenda ku dduuka.", "Enkya yagenda ku dduuka.", "Tomorrow he/she will go to the shop.", "-naa- → -a-", "", "high"),
    ("Enkya banaagenda mu katale.", "Enkya baagenda mu katale.", "Tomorrow they will go to the market.", "-naa- → -a-", "", "high"),
    ("Enkya tunaasoma ekitabo.", "Enkya twasoma ekitabo.", "Tomorrow we will read a book.", "-naa- → -a-", "", "high"),
    ("Enkya anaalima ennimiro.", "Enkya yalima ennimiro.", "Tomorrow he/she will dig the garden.", "-naa- → -a-", "", "high"),
    ("Enkya banaakola nnyo.", "Enkya baakola nnyo.", "Tomorrow they will work hard.", "-naa- → -a-", "", "high"),
    ("Enkya nnaagula engoye.", "Enkya nnagula engoye.", "Tomorrow I will buy clothes.", "-naa- → -a-", "", "high"),
    ("Enkya anaatuuka ku makya.", "Enkya yatuuka ku makya.", "Tomorrow he/she will arrive in the morning.", "-naa- → -a-", "", "high"),
    ("Enkya banaazina.", "Enkya baazina.", "Tomorrow they will dance.", "-naa- → -a-", "", "high"),
    ("Enkya tunaayimba mu kkanisa.", "Enkya twayimba mu kkanisa.", "Tomorrow we will sing in church.", "-naa- → -a-", "", "high"),
    ("Enkya onoogenda ku ssomero.", "Enkya wagenda ku ssomero.", "Tomorrow you (sg.) will go to school.", "-noo- → -a- (2sg forms)", "Check the 2sg forms onoo- / wa-.", "medium"),
    # omwaka oguwedde (last year): past is correct, far future is wrong
    ("Omwaka oguwedde baalima kasooli.", "Omwaka oguwedde balirima kasooli.", "Last year they grew maize.", "-a- → -li-", "", "high"),
    ("Omwaka oguwedde twazimba ennyumba.", "Omwaka oguwedde tulizimba ennyumba.", "Last year we built a house.", "-a- → -li-", "", "high"),
    ("Omwaka oguwedde yasoma e Makerere.", "Omwaka oguwedde alisoma e Makerere.", "Last year he/she studied at Makerere.", "-a- → -li-", "", "high"),
    ("Omwaka oguwedde nnagula emmotoka.", "Omwaka oguwedde ndigula emmotoka.", "Last year I bought a car.", "-a- → -li-", "", "high"),
    ("Omwaka oguwedde baatuuka e Bungereza.", "Omwaka oguwedde balituuka e Bungereza.", "Last year they arrived in England.", "-a- → -li-", "", "high"),
    ("Omwaka oguwedde twafuna enkuba nnyingi.", "Omwaka oguwedde tulifuna enkuba nnyingi.", "Last year we got a lot of rain.", "-a- → -li-", "", "high"),
    ("Omwaka oguwedde yakola mu bbanka.", "Omwaka oguwedde alikola mu bbanka.", "Last year he/she worked in a bank.", "-a- → -li-", "", "high"),
    ("Omwaka oguwedde nnayiga okuvuga.", "Omwaka oguwedde ndiyiga okuvuga.", "Last year I learned to drive.", "-a- → -li-", "", "high"),
    # omwaka ogujja (next year): far future is correct, past is wrong
    ("Omwaka ogujja balirima kasooli.", "Omwaka ogujja baalima kasooli.", "Next year they will grow maize.", "-li- → -a- (far future → past)", "", "high"),
    ("Omwaka ogujja tulizimba ennyumba.", "Omwaka ogujja twazimba ennyumba.", "Next year we will build a house.", "-li- → -a-", "", "high"),
    ("Omwaka ogujja alisoma e Makerere.", "Omwaka ogujja yasoma e Makerere.", "Next year he/she will study at Makerere.", "-li- → -a-", "", "high"),
    ("Omwaka ogujja ndigula emmotoka.", "Omwaka ogujja nnagula emmotoka.", "Next year I will buy a car.", "-li- → -a-", "", "high"),
    ("Omwaka ogujja balituuka e Bungereza.", "Omwaka ogujja baatuuka e Bungereza.", "Next year they will arrive in England.", "-li- → -a-", "", "high"),
    ("Omwaka ogujja tulifuna enkuba nnyingi.", "Omwaka ogujja twafuna enkuba nnyingi.", "Next year we will get a lot of rain.", "-li- → -a-", "", "high"),
    ("Omwaka ogujja alikola mu bbanka.", "Omwaka ogujja yakola mu bbanka.", "Next year he/she will work in a bank.", "-li- → -a-", "", "high"),
    ("Omwaka ogujja ndiyiga okuvuga.", "Omwaka ogujja nnayiga okuvuga.", "Next year I will learn to drive.", "-li- → -a-", "", "high"),
    # wiiki ewedde / wiiki ejja (last week / next week)
    ("Wiiki ewedde twalaba omupiira.", "Wiiki ewedde tuliraba omupiira.", "Last week we watched football.", "-a- → -li-", "", "high"),
    ("Wiiki ewedde baagenda e Mbarara.", "Wiiki ewedde baligenda e Mbarara.", "Last week they went to Mbarara.", "-a- → -li-", "", "high"),
    ("Wiiki ewedde yafumba emmere ennungi.", "Wiiki ewedde alifumba emmere ennungi.", "Last week he/she cooked good food.", "-a- → -li-", "", "high"),
    ("Wiiki ewedde nnawandiika ebbaluwa.", "Wiiki ewedde ndiwandiika ebbaluwa.", "Last week I wrote a letter.", "-a- → -li-", "", "high"),
    ("Wiiki ewedde baakyala ewaffe.", "Wiiki ewedde balikyala ewaffe.", "Last week they visited our home.", "-a- → -li-", "", "medium"),
    ("Wiiki ejja tuliraba omupiira.", "Wiiki ejja twalaba omupiira.", "Next week we will watch football.", "-li- → -a-", "", "high"),
    ("Wiiki ejja baligenda e Mbarara.", "Wiiki ejja baagenda e Mbarara.", "Next week they will go to Mbarara.", "-li- → -a-", "", "high"),
    ("Wiiki ejja alifumba emmere ennungi.", "Wiiki ejja yafumba emmere ennungi.", "Next week he/she will cook good food.", "-li- → -a-", "", "high"),
    ("Wiiki ejja ndiwandiika ebbaluwa.", "Wiiki ejja nnawandiika ebbaluwa.", "Next week I will write a letter.", "-li- → -a-", "", "high"),
    ("Wiiki ejja balikyala ewaffe.", "Wiiki ejja baakyala ewaffe.", "Next week they will visit our home.", "-li- → -a-", "", "medium"),
]

NEGATION = [
    # -nnaba (not yet) exists only in the negative
    ("Omwana tannaba kulya.", "Omwana annaba kulya.", "The child has not yet eaten.", "ta- → a- (negation removed)", "", "high"),
    ("Abagenyi tebannaba kutuuka.", "Abagenyi bannaba kutuuka.", "The guests have not yet arrived.", "teba- → ba-", "", "high"),
    ("Enkuba tennaba kutonnya.", "Enkuba ennaba kutonnya.", "The rain has not yet fallen.", "te- → e-", "", "high"),
    ("Sinnaba kusoma kitabo kino.", "Nnaba kusoma kitabo kino.", "I have not yet read this book.", "si- → n-", "kitabo has no augment because the verb is negative.", "high"),
    ("Tetunnaba kugenda.", "Tunnaba kugenda.", "We have not yet gone.", "tetu- → tu-", "", "high"),
    ("Emmere tennaba kuggya.", "Emmere ennaba kuggya.", "The food is not yet cooked.", "te- → e-", "Check the verb -ggya (be cooked).", "medium"),
    ("Omusomesa tannaba kujja.", "Omusomesa annaba kujja.", "The teacher has not yet come.", "ta- → a-", "", "high"),
    ("Abaana tebannaba kwebaka.", "Abaana bannaba kwebaka.", "The children have not yet slept.", "teba- → ba-", "", "high"),
    ("Ekitabo tekinnaba kutuuka.", "Ekitabo kinnaba kutuuka.", "The book has not yet arrived.", "teki- → ki-", "", "high"),
    ("Bbaasi tennaba kuva wano.", "Bbaasi ennaba kuva wano.", "The bus has not yet left here.", "te- → e-", "", "high"),
    ("Tonnaba kumaliriza mulimu.", "Onnaba kumaliriza mulimu.", "You (sg.) have not yet finished the work.", "to- → o-", "", "medium"),
    ("Temunnaba kusasula.", "Munnaba kusasula.", "You (pl.) have not yet paid.", "temu- → mu-", "", "high"),
    ("Abalimi tebannaba kukungula.", "Abalimi bannaba kukungula.", "The farmers have not yet harvested.", "teba- → ba-", "", "high"),
    ("Omulwadde tannaba kuwona.", "Omulwadde annaba kuwona.", "The patient has not yet recovered.", "ta- → a-", "", "high"),
    ("Ebbaluwa tennaba kutuuka.", "Ebbaluwa ennaba kutuuka.", "The letter has not yet arrived.", "te- → e-", "", "high"),
    # a noun without the augment is correct only after a negative verb
    ("Sirina kitabo.", "Nnina kitabo.", "I do not have a book.", "si- → n- (negation removed)", "The positive needs the augment: Nnina ekitabo.", "high"),
    ("Talina mmere.", "Alina mmere.", "He/she has no food.", "ta- → a-", "Positive: Alina emmere.", "high"),
    ("Tebalina sente.", "Balina sente.", "They have no money.", "teba- → ba-", "Positive: Balina ensimbi / sente (check).", "medium"),
    ("Tetulina mazzi.", "Tulina mazzi.", "We have no water.", "tetu- → tu-", "Positive: Tulina amazzi.", "high"),
    ("Saalaba muntu.", "Nnalaba muntu.", "I did not see anyone.", "saa- → nna- (negative past → past)", "Positive: Nnalaba omuntu.", "high"),
    ("Teyalya mmere.", "Yalya mmere.", "He/she did not eat food.", "teya- → ya-", "Positive: Yalya emmere.", "high"),
    ("Tebaalaba mbwa.", "Baalaba mbwa.", "They did not see a dog.", "tebaa- → baa-", "Positive: Baalaba embwa.", "high"),
    ("Sirina mwana.", "Nnina mwana.", "I have no child.", "si- → n-", "Positive: Nnina omwana.", "high"),
    ("Talina mulimu.", "Alina mulimu.", "He/she has no job.", "ta- → a-", "Positive: Alina omulimu.", "high"),
    ("Tetwagula mmotoka.", "Twagula mmotoka.", "We did not buy a car.", "tetwa- → twa-", "Positive: Twagula emmotoka.", "high"),
    ("Sinywa mwenge.", "Nnywa mwenge.", "I do not drink beer.", "si- → n-", "Positive: Nnywa omwenge.", "high"),
    ("Talya nnyama.", "Alya nnyama.", "He/she does not eat meat.", "ta- → a-", "Positive: Alya ennyama.", "high"),
    ("Tebasoma bitabo.", "Basoma bitabo.", "They do not read books.", "teba- → ba-", "Positive: Basoma ebitabo.", "high"),
    ("Tetulaba kintu.", "Tulaba kintu.", "We do not see anything.", "tetu- → tu-", "Positive: Tulaba ekintu.", "high"),
    ("Omwana talina ngatto.", "Omwana alina ngatto.", "The child has no shoes.", "ta- → a-", "Positive: Omwana alina engatto.", "high"),
    # negative polarity items (n'omu, n'akatono, n'ekimu) need a negative verb
    ("Sirina mwana n'omu.", "Nnina mwana n'omu.", "I do not have even one child.", "si- → n-", "", "high"),
    ("Tebalina sente n'akatono.", "Balina sente n'akatono.", "They have no money at all.", "teba- → ba-", "", "high"),
    ("Talina kitabo n'ekimu.", "Alina kitabo n'ekimu.", "He/she does not have even one book.", "ta- → a-", "", "high"),
    ("Saalaba muntu n'omu.", "Nnalaba muntu n'omu.", "I did not see even one person.", "saa- → nna-", "", "high"),
    ("Tetulina budde n'akatono.", "Tulina budde n'akatono.", "We have no time at all.", "tetu- → tu-", "", "high"),
    # the negative prefix agrees with the subject
    ("Omwana tagenda ku ssomero.", "Omwana tebagenda ku ssomero.", "The child does not go to school.", "ta- → teba- (neg cl1 → cl2)", "Negation and subject marker are one prefix.", "high"),
    ("Abaana tebagenda ku ssomero.", "Abaana tagenda ku ssomero.", "The children do not go to school.", "teba- → ta-", "", "high"),
    ("Nze sigenda ku ssomero.", "Nze tagenda ku ssomero.", "I do not go to school.", "si- → ta- (neg 1sg → 3sg)", "", "high"),
    ("Ffe tetugenda ku ssomero.", "Ffe tebagenda ku ssomero.", "We do not go to school.", "tetu- → teba-", "", "high"),
    ("Ggwe togenda ku ssomero.", "Ggwe tagenda ku ssomero.", "You (sg.) do not go to school.", "to- → ta-", "", "high"),
    ("Mmwe temugenda ku ssomero.", "Mmwe tetugenda ku ssomero.", "You (pl.) do not go to school.", "temu- → tetu-", "", "high"),
    ("Embwa terya nnyama.", "Embwa talya nnyama.", "The dog does not eat meat.", "te- → ta- (neg cl9 → cl1)", "", "medium"),
    ("Ekitabo tekiri wano.", "Ekitabo tali wano.", "The book is not here.", "teki- → ta-", "", "high"),
    ("Ebitabo tebiri wano.", "Ebitabo tekiri wano.", "The books are not here.", "tebi- → teki-", "", "high"),
    ("Omuti teguli wano.", "Omuti tekiri wano.", "The tree is not here.", "tegu- → teki-", "", "medium"),
    ("Ente teziri mu nnimiro.", "Ente tebali mu nnimiro.", "The cows are not in the field.", "tezi- → teba-", "", "high"),
    ("Omuyizi tali mu kibiina.", "Omuyizi tebali mu kibiina.", "The student is not in the classroom.", "ta- → teba-", "Check that no honorific plural reading is possible.", "medium"),
    ("Abayizi tebali mu kibiina.", "Abayizi tali mu kibiina.", "The students are not in the classroom.", "teba- → ta-", "", "high"),
    ("Amazzi tegali mu nsuwa.", "Amazzi tekiri mu nsuwa.", "The water is not in the pot.", "tega- → teki-", "", "medium"),
    ("Olugoye teluli wano.", "Olugoye tekiri wano.", "The cloth is not here.", "telu- → teki-", "", "high"),
]

# (consistent, switched, referent, class correct -> wrong, english, note, confidence)
PARAGRAPHS = [
    ("Kato yaguze ekitabo ekiggya. Yakisomye mu lunaku lumu. Kati kiri ku mmeeza.", "Kato yaguze ekitabo ekiggya. Yakisomye mu lunaku lumu. Kati giri ku mmeeza.", "ekitabo", "cl7 → cl9 (ki- → gi-)", "Kato bought a new book. He read it in one day. Now it is on the table.", "", "high"),
    ("Musoke alina embwa ennene. Buli kiro ekuuma ennyumba ye. Abaana bagyagala nnyo.", "Musoke alina embwa ennene. Buli kiro ekuuma ennyumba ye. Abaana bakyagala nnyo.", "embwa", "cl9 → cl7 (-gi- → -ki-)", "Musoke has a big dog. Every night it guards his house. The children love it very much.", "", "high"),
    ("Mu luggya lwaffe mulimu omuti omunene. Gulina ebibala bingi. Abaana batera okugulinnya.", "Mu luggya lwaffe mulimu omuti omunene. Gulina ebibala bingi. Abaana batera okukilinnya.", "omuti", "cl3 → cl7 (-gu- → -ki-)", "In our courtyard there is a big tree. It has many fruits. The children often climb it.", "", "high"),
    ("Taata yaguze emmotoka empya. Ya langi enjeru. Buli ku makya agivuga okugenda ku mulimu.", "Taata yaguze emmotoka empya. Ya langi enjeru. Buli ku makya akivuga okugenda ku mulimu.", "emmotoka", "cl9 → cl7 (-gi- → -ki-)", "Father bought a new car. It is white. Every morning he drives it to work.", "", "high"),
    ("Abaana bazze okuva ku ssomero. Balumwa enjala nnyo. Maama abawadde emmere.", "Abaana bazze okuva ku ssomero. Balumwa enjala nnyo. Maama amuwadde emmere.", "abaana", "cl2 → cl1 (-ba- → -mu-)", "The children have come back from school. They are very hungry. Mother has given them food.", "", "high"),
    ("Omwana yagudde ku kkubo. Yakaaba nnyo. Nnyina yamusitudde.", "Omwana yagudde ku kkubo. Yakaaba nnyo. Nnyina yakisitudde.", "omwana", "cl1 → cl7 (-mu- → -ki-)", "The child fell on the road. He/she cried a lot. His/her mother picked him/her up.", "", "high"),
    ("Twagula amazzi mu dduuka. Gaali manyogovu. Twaganywa mangu.", "Twagula amazzi mu dduuka. Gaali manyogovu. Twakinywa mangu.", "amazzi", "cl6 → cl7 (-ga- → -ki-)", "We bought water in the shop. It was cold. We drank it quickly.", "", "medium"),
    ("Kampala kye kibuga ekikulu kya Uganda. Kirimu abantu bangi. Abagenyi bangi bakyagala.", "Kampala kye kibuga ekikulu kya Uganda. Kirimu abantu bangi. Abagenyi bangi bagyagala.", "ekibuga", "cl7 → cl9 (-ki- → -gi-)", "Kampala is the capital city of Uganda. It has many people. Many visitors love it.", "", "high"),
    ("Bazadde bange bazimbye ennyumba empya. Erina ebisenge bina. Buli muntu agyagala.", "Bazadde bange bazimbye ennyumba empya. Erina ebisenge bina. Buli muntu akyagala.", "ennyumba", "cl9 → cl7 (-gi- → -ki-)", "My parents have built a new house. It has four rooms. Everybody likes it.", "", "high"),
    ("Nnaguze olugoye olulungi. Lwa langi emmyufu. Nja kulwambala ku Ssande.", "Nnaguze olugoye olulungi. Lwa langi emmyufu. Nja kukyambala ku Ssande.", "olugoye", "cl11 → cl7 (-lu- → -ki-)", "I bought a nice cloth. It is red. I will wear it on Sunday.", "", "medium"),
    ("Taata yasimbye emiti mingi. Kati giwanvu nnyo. Tugifukirira buli lunaku.", "Taata yasimbye emiti mingi. Kati giwanvu nnyo. Tugufukirira buli lunaku.", "emiti", "cl4 → cl3 (-gi- → -gu-)", "Father planted many trees. Now they are very tall. We water them every day.", "", "high"),
    ("Omusomesa yatuwadde ebitabo bisatu. Biwandiikiddwa mu Luganda. Tubisoma buli lunaku.", "Omusomesa yatuwadde ebitabo bisatu. Biwandiikiddwa mu Luganda. Tukisoma buli lunaku.", "ebitabo", "cl8 → cl7 (-bi- → -ki-)", "The teacher gave us three books. They are written in Luganda. We read them every day.", "", "high"),
    ("Maama alina akambe akasongovu. Kasala bulungi nnyo. Akakozesa okusala enva.", "Maama alina akambe akasongovu. Kasala bulungi nnyo. Akikozesa okusala enva.", "akambe", "cl12 → cl7 (-ka- → -ki-)", "Mother has a sharp knife. It cuts very well. She uses it to cut vegetables.", "", "medium"),
    ("Jjajja alina ente nnyingi. Zirya omuddo mu nnimiro. Buli kawungeezi azikomyawo eka.", "Jjajja alina ente nnyingi. Zirya omuddo mu nnimiro. Buli kawungeezi agikomyawo eka.", "ente (plural)", "cl10 → cl9 (-zi- → -gi-)", "Grandparent has many cows. They eat grass in the field. Every evening he/she brings them home.", "This switch changes number (plural → singular) in the same noun class pair.", "medium"),
    ("Mu kyalo kyaffe mulimu omugga omunene. Gukulukuta okuva ku lusozi. Abantu bagukozesa okunaaba.", "Mu kyalo kyaffe mulimu omugga omunene. Gukulukuta okuva ku lusozi. Abantu bakikozesa okunaaba.", "omugga", "cl3 → cl7 (-gu- → -ki-)", "In our village there is a big river. It flows from the hill. People use it for bathing.", "", "medium"),
    ("Nnaguze amagi kkumi mu katale. Gaali mapya. Maama yagafumbye ku makya.", "Nnaguze amagi kkumi mu katale. Gaali mapya. Maama yagifumbye ku makya.", "amagi", "cl6 → cl9 (-ga- → -gi-)", "I bought ten eggs in the market. They were fresh. Mother cooked them in the morning.", "", "medium"),
    ("Ku kkubo waliwo ejjinja eddene. Lizibye ekkubo. Abasajja balisitudde.", "Ku kkubo waliwo ejjinja eddene. Lizibye ekkubo. Abasajja bakisitudde.", "ejjinja", "cl5 → cl7 (-li- → -ki-)", "On the road there is a big stone. It has blocked the road. The men have lifted it.", "balisitudde could also be read as a future form; check.", "low"),
    ("Omwana yamenye ekikopo. Kyali kya maama. Kati kiri mu bitundu bibiri.", "Omwana yamenye ekikopo. Kyali kya maama. Kati guli mu bitundu bibiri.", "ekikopo", "cl7 → cl3 (ki- → gu-)", "The child broke the cup. It was mother's. Now it is in two pieces.", "", "high"),
    ("Abagenyi batuuse ku ssaawa ttaano. Baali bakooye nnyo. Twabawadde amazzi okunywa.", "Abagenyi batuuse ku ssaawa ttaano. Baali bakooye nnyo. Twamuwadde amazzi okunywa.", "abagenyi", "cl2 → cl1 (-ba- → -mu-)", "The guests arrived at eleven o'clock. They were very tired. We gave them water to drink.", "", "high"),
    ("Kizza yaguze embuzi mu katale. Erina amaaso amaddugavu. Agisibye wansi w'omuti.", "Kizza yaguze embuzi mu katale. Erina amaaso amaddugavu. Akisibye wansi w'omuti.", "embuzi", "cl9 → cl7 (-gi- → -ki-)", "Kizza bought a goat in the market. It has black eyes. He has tied it under a tree.", "", "high"),
    ("Abaana baguze omupiira omuggya. Gwa langi ya kyenvu. Bagusambye olunaku lwonna.", "Abaana baguze omupiira omuggya. Gwa langi ya kyenvu. Bakisambye olunaku lwonna.", "omupiira", "cl3 → cl7 (-gu- → -ki-)", "The children bought a new ball. It is yellow. They have kicked it all day.", "", "medium"),
    ("Essomero lyaffe liri ku lusozi. Lirina abayizi bangi. Lyatandika emyaka mingi emabega.", "Essomero lyaffe liri ku lusozi. Lirina abayizi bangi. Kyatandika emyaka mingi emabega.", "essomero", "cl5 → cl7 (lya- → kya-)", "Our school is on the hill. It has many students. It started many years ago.", "", "medium"),
    ("Ku mugga waliwo olutindo oluggya. Luzimbiddwa mu kyuma. Abantu bangi baluyitako buli lunaku.", "Ku mugga waliwo olutindo oluggya. Luzimbiddwa mu kyuma. Abantu bangi bakiyitako buli lunaku.", "olutindo", "cl11 → cl7 (-lu- → -ki-)", "On the river there is a new bridge. It is built of iron. Many people cross it every day.", "", "medium"),
    ("Nnalabye ekinyonyi ekirungi ku muti. Kyali kya langi ya bbululu. Kyabuuse mangu ddala.", "Nnalabye ekinyonyi ekirungi ku muti. Kyali kya langi ya bbululu. Gwabuuse mangu ddala.", "ekinyonyi", "cl7 → cl3 (kya- → gwa-)", "I saw a beautiful bird on a tree. It was blue. It flew away very quickly.", "", "medium"),
    ("Maama yafumbye amatooke. Gaali gawooma nnyo. Twagalidde ku kyeggulo.", "Maama yafumbye amatooke. Gaali gawooma nnyo. Twakiridde ku kyeggulo.", "amatooke", "cl6 → cl7 (-ga- → -ki-)", "Mother cooked matooke. It was very tasty. We ate it at supper.", "", "medium"),
    ("Nnafunye ebbaluwa okuva ewa mukwano gwange. Yali mpanvu nnyo. Nnagisomye emirundi ebiri.", "Nnafunye ebbaluwa okuva ewa mukwano gwange. Yali mpanvu nnyo. Nnakisomye emirundi ebiri.", "ebbaluwa", "cl9 → cl7 (-gi- → -ki-)", "I received a letter from my friend. It was very long. I read it two times.", "", "high"),
    ("Abayizi bayize oluyimba oluggya. Lunyuma nnyo. Baluyimba buli ku makya.", "Abayizi bayize oluyimba oluggya. Lunyuma nnyo. Bakiyimba buli ku makya.", "oluyimba", "cl11 → cl7 (-lu- → -ki-)", "The students have learned a new song. It is very nice. They sing it every morning.", "", "medium"),
    ("Jjajja abeera mu kyalo ekitono. Kiri wala okuva e Kampala. Tukikyalira buli Ssekukkulu.", "Jjajja abeera mu kyalo ekitono. Kiri wala okuva e Kampala. Tugukyalira buli Ssekukkulu.", "ekyalo", "cl7 → cl3 (-ki- → -gu-)", "Grandparent lives in a small village. It is far from Kampala. We visit it every Christmas.", "", "medium"),
    ("Omusomesa waffe mulungi nnyo. Atusomesa okubala. Abayizi bonna bamwagala.", "Omusomesa waffe mulungi nnyo. Atusomesa okubala. Abayizi bonna bakyagala.", "omusomesa", "cl1 → cl7 (-mu- → -ki-)", "Our teacher is very good. He/she teaches us mathematics. All the students love him/her.", "", "high"),
    ("Maama yafumbye emmere nnyingi. Yali ewooma nnyo. Abagenyi bonna baagiridde.", "Maama yafumbye emmere nnyingi. Yali ewooma nnyo. Abagenyi bonna baakiridde.", "emmere", "cl9 → cl7 (-gi- → -ki-)", "Mother cooked a lot of food. It was very tasty. All the guests ate it.", "", "medium"),
]

# (form a, form b, english, type of variation, note, confidence)
VARIATION = [
    ("Nze ŋŋenda ku ssomero.", "Nze ngenda ku ssomero.", "I am going to school.", "spelling (ŋŋ / ng)", "", "high"),
    ("Weebale nnyo.", "Webale nnyo.", "Thank you very much.", "spelling (ee / e)", "", "high"),
    ("Maama agenda mu katale.", "Maama bagenda mu katale.", "Mother is going to the market.", "honorific plural", "", "high"),
    ("Nja kugenda e Kampala.", "Nnaagenda e Kampala.", "I will go to Kampala.", "future form (ja ku- / -naa-)", "", "high"),
    ("Ennyumba yange nnene.", "Enju yange nnene.", "My house is big.", "synonym (ennyumba / enju)", "", "high"),
    ("Tewali kizibu.", "Tewali buzibu.", "There is no problem.", "noun class of the noun (ki- / bu-)", "", "medium"),
    ("Simanyi.", "Sikimanyi.", "I do not know.", "optional object marker", "", "medium"),
    ("Atuuse.", "Yatuuse.", "He/she has arrived.", "perfect / recent past", "", "medium"),
    ("Nnalaba omwana.", "Omwana nnamulaba.", "I saw the child.", "word order with object marker", "", "high"),
    ("Kale.", "Kaale.", "Okay.", "spelling (a / aa)", "", "medium"),
    ("Nnina ssente.", "Nnina sente.", "I have money.", "spelling (ss / s)", "", "high"),
    ("Ali e Kampala.", "Ali mu Kampala.", "He/she is in Kampala.", "locative (e / mu)", "", "medium"),
    ("Jjajja ajja enkya.", "Jjajja bajja enkya.", "Grandparent is coming tomorrow.", "honorific plural", "", "high"),
    ("Omusajja gwe nnalaba yagenda.", "Omusajja nnalaba yagenda.", "The man that I saw went away.", "optional relative pronoun", "Not sure that the short form is accepted.", "low"),
    ("Yali agenda.", "Yali ng'agenda.", "He/she was going.", "progressive form", "", "medium"),
    ("Webale okujja.", "Webale kujja.", "Thank you for coming.", "optional augment on the infinitive", "", "medium"),
    ("Weebale ssebo.", "Weebale sebo.", "Thank you, sir.", "spelling (ss / s)", "", "high"),
    ("Leero enkuba etonnya.", "Enkuba etonnya leero.", "Today it is raining.", "word order", "", "high"),
    ("Nnyabo, weebale.", "Nyabo, weebale.", "Madam, thank you.", "spelling (nny / ny)", "", "high"),
    ("Taata yagenze ku mulimu.", "Taata baagenze ku mulimu.", "Father has gone to work.", "honorific plural", "", "high"),
]

ARIAL = "Arial"
HEAD_FILL = PatternFill("solid", start_color="1F3864")
INPUT_FILL = PatternFill("solid", start_color="FFFF00")
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
STATUSES = ["TODO", "OK", "FIXED", "DROP"]


def style_sheet(ws, headers, widths, input_cols, n_rows):
    for c, (h, w) in enumerate(zip(headers, widths), start=1):
        cell = ws.cell(row=1, column=c, value=h)
        cell.font = Font(name=ARIAL, bold=True, color="FFFFFF")
        cell.fill = HEAD_FILL
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        cell.border = BORDER
        ws.column_dimensions[get_column_letter(c)].width = w
    for r in range(2, n_rows + 2):
        for c in range(1, len(headers) + 1):
            cell = ws.cell(row=r, column=c)
            cell.font = Font(name=ARIAL)
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            cell.border = BORDER
            if headers[c - 1] in input_cols:
                cell.fill = INPUT_FILL
    ws.freeze_panes = "C2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{n_rows + 1}"
    status_col = get_column_letter(headers.index("STATUS") + 1)
    dv = DataValidation(type="list", formula1='"' + ",".join(STATUSES) + '"', allow_blank=False)
    dv.error = "Use TODO, OK, FIXED or DROP."
    dv.prompt = "OK = correct as written. FIXED = you wrote a correction. DROP = do not use."
    ws.add_data_validation(dv)
    dv.add(f"{status_col}2:{status_col}{n_rows + 1}")


def build(path):
    wb = Workbook()

    readme = wb.active
    readme.title = "README"
    lines = [
        ("Luganda evaluation items: draft for native-speaker validation", True),
        ("", False),
        ("Claude wrote all items in this workbook. Claude is not a native speaker, so the items can contain errors.", False),
        ("The evaluation notebooks use only the rows with STATUS = OK or FIXED.", False),
        ("", False),
        ("How to validate", True),
        ("1. Read each row. Edit only the yellow columns.", False),
        ("2. Set STATUS: OK = the row is correct as written. FIXED = you wrote a correction. DROP = do not use the row.", False),
        ("3. For FIXED: write the corrected text in the fixed_* columns. Leave a fixed_* column empty if that part is correct.", False),
        ("4. For minimal pairs: the correct and the wrong sentence must differ in ONE morpheme only. The wrong sentence must be ungrammatical.", False),
        ("5. Use validator_note for any comment (for example, a dialect form, or a second correct reading).", False),
        ("6. Do not change the id column. Do not rename the sheets or the column headers.", False),
        ("", False),
        ("Columns", True),
        ("my_confidence: how sure Claude is about the Luganda (high / medium / low). Check the low and medium rows first.", False),
        ("changed_morpheme: the morpheme that is different in the wrong sentence.", False),
        ("Example row: NC-001. Correct: 'Ekitabo ekinene kiri ku mmeeza.' Wrong: 'Ekitabo ennene kiri ku mmeeza.' The adjective has the cl9 prefix, but ekitabo is cl7.", False),
        ("", False),
        ("Sheets", True),
        ("minimal_pairs: 200 pairs, 50 for each error type (noun_class_agreement, subject_object_marker, tense_aspect, negation).", False),
        ("paragraphs: 30 short paragraphs. The switched version uses a different noun class for the same referent in the last sentence.", False),
        ("variation: 20 items with two forms that a speaker accepts. The model must not mark either form as wrong.", False),
        ("progress: counts of STATUS for each sheet and each error type (formulas, updated automatically).", False),
    ]
    for i, (text, bold) in enumerate(lines, start=1):
        cell = readme.cell(row=i, column=1, value=text)
        cell.font = Font(name=ARIAL, bold=bold, size=13 if i == 1 else 11)
        cell.alignment = Alignment(wrap_text=True)
    readme.column_dimensions["A"].width = 120

    mp = wb.create_sheet("minimal_pairs")
    headers = ["id", "error_type", "correct", "wrong", "english", "changed_morpheme", "note", "my_confidence",
               "STATUS", "fixed_correct", "fixed_wrong", "validator_note"]
    mp.append(headers)
    rows = []
    for prefix, etype, items in [("NC", "noun_class_agreement", NOUN_CLASS), ("SO", "subject_object_marker", SUBJ_OBJ),
                                 ("TA", "tense_aspect", TENSE), ("NG", "negation", NEGATION)]:
        assert len(items) == 50, (etype, len(items))
        for k, (cor, wrg, eng, morph, note, conf) in enumerate(items, start=1):
            assert cor != wrg
            rows.append([f"{prefix}-{k:03d}", etype, cor, wrg, eng, morph, note, conf, "TODO", "", "", ""])
    for r in rows:
        mp.append(r)
    style_sheet(mp, headers, [9, 20, 38, 38, 34, 26, 30, 11, 10, 34, 34, 30],
                {"STATUS", "fixed_correct", "fixed_wrong", "validator_note"}, len(rows))

    pg = wb.create_sheet("paragraphs")
    headers_p = ["id", "consistent", "switched", "referent", "class_switch", "english", "note", "my_confidence",
                 "STATUS", "fixed_consistent", "fixed_switched", "validator_note"]
    pg.append(headers_p)
    assert len(PARAGRAPHS) == 30
    for k, (con, sw, ref, cls, eng, note, conf) in enumerate(PARAGRAPHS, start=1):
        assert con != sw
        pg.append([f"PG-{k:03d}", con, sw, ref, cls, eng, note, conf, "TODO", "", "", ""])
    style_sheet(pg, headers_p, [9, 50, 50, 14, 22, 45, 28, 11, 10, 45, 45, 30],
                {"STATUS", "fixed_consistent", "fixed_switched", "validator_note"}, len(PARAGRAPHS))

    va = wb.create_sheet("variation")
    headers_v = ["id", "form_a", "form_b", "english", "variation_type", "note", "my_confidence",
                 "STATUS", "fixed_form_a", "fixed_form_b", "validator_note"]
    va.append(headers_v)
    assert len(VARIATION) == 20
    for k, (a, b, eng, vt, note, conf) in enumerate(VARIATION, start=1):
        va.append([f"VA-{k:03d}", a, b, eng, vt, note, conf, "TODO", "", "", ""])
    style_sheet(va, headers_v, [9, 36, 36, 32, 28, 28, 11, 10, 36, 36, 30],
                {"STATUS", "fixed_form_a", "fixed_form_b", "validator_note"}, len(VARIATION))

    pr = wb.create_sheet("progress")
    pr.append(["sheet / error type", "total", "TODO", "OK", "FIXED", "DROP", "usable (OK + FIXED)"])
    groups = [("minimal_pairs: noun_class_agreement", "minimal_pairs", "noun_class_agreement"),
              ("minimal_pairs: subject_object_marker", "minimal_pairs", "subject_object_marker"),
              ("minimal_pairs: tense_aspect", "minimal_pairs", "tense_aspect"),
              ("minimal_pairs: negation", "minimal_pairs", "negation"),
              ("paragraphs", "paragraphs", None), ("variation", "variation", None)]
    for i, (label, sheet, etype) in enumerate(groups, start=2):
        status_col = {"minimal_pairs": "I", "paragraphs": "I", "variation": "H"}[sheet]
        n = {"minimal_pairs": 201, "paragraphs": 31, "variation": 21}[sheet]
        rng = f"{sheet}!${status_col}$2:${status_col}${n}"
        pr.cell(row=i, column=1, value=label)
        if etype:
            crit = f'{sheet}!$B$2:$B${n},"{etype}"'
            pr.cell(row=i, column=2, value=f"=COUNTIFS({crit})")
            for c, s in enumerate(STATUSES, start=3):
                pr.cell(row=i, column=c, value=f'=COUNTIFS({crit},{rng},"{s}")')
        else:
            pr.cell(row=i, column=2, value=f'=COUNTA({sheet}!$A$2:$A${n})')
            for c, s in enumerate(STATUSES, start=3):
                pr.cell(row=i, column=c, value=f'=COUNTIF({rng},"{s}")')
        pr.cell(row=i, column=7, value=f"=D{i}+E{i}")
    last = len(groups) + 1
    pr.cell(row=last + 1, column=1, value="all")
    for c in range(2, 8):
        col = get_column_letter(c)
        pr.cell(row=last + 1, column=c, value=f"=SUM({col}2:{col}{last})")
    for r in range(1, last + 2):
        for c in range(1, 8):
            cell = pr.cell(row=r, column=c)
            cell.font = Font(name=ARIAL, bold=(r == 1 or r == last + 1), color="FFFFFF" if r == 1 else "000000")
            cell.border = BORDER
            if r == 1:
                cell.fill = HEAD_FILL
    pr.column_dimensions["A"].width = 40
    for c in "BCDEFG":
        pr.column_dimensions[c].width = 14
    pr["A1"].comment = Comment("Counts update when you change STATUS in the other sheets.", "Claude")

    wb.save(path)
    print("wrote", path, "minimal pairs:", len(rows))


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "data/luganda_eval_items.xlsx")
