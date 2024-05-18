import torch

from Optimizer import Bundle
from Tensorvec import Vec
from toolfunc import Ut
import random,time,math
from tqdm import tqdm

def generate(words, bondlist):
    Losss = {}
    for word in bondlist.keys():
        lossSum = 0
        if any(word == wds.val() for wds in words):
            continue
        for wd in range(len(words)):
            lossSum += torch.sqrt(words[wd](bondlist[word], optim=False,distance=len(words)-wd))
        Losss[word] = lossSum

    if len(Losss) == 1:
        return list(Losss.keys())[0]
    key = random.choices(list(Losss.keys()), weights=list(map(lambda x:max(Losss.values())-x if x != 0 else 0,Losss.values())), k=1)[0]
    return key











if __name__ == '__main__':

    trainTest = ['''Despite the relentless march of time, which ceaselessly ushers in the new while sweeping away the old, the enduring beauty of the verdant hills, bathed in the golden glow of the setting sun, remains a constant source of inspiration for the humble poet, who, armed with nothing more than a quill and a piece of parchment, seeks to capture the fleeting moments of life in the indelible ink of his words, weaving together a tapestry of human experience that transcends the mundane and touches the sublime, a testament to the indomitable spirit of mankind in the face of the inexorable passage of time.''',
                 'Life is too short to spend time with people who suck the happiness out of you. If someone wants you in their life, they’ll make room for you. You shouldn’t have to fight for a spot. Never, ever insist yourself to someone who continuously overlooks your worth. And remember, it’s not the people that stand by your side when you’re at your best, but the ones who stand beside you when you’re at your worst that are your true friends.',
                 'In the flood of darkness, hope is the light. It brings comfort, faith, and confidence. It gives us guidance when we are lost, and gives support when we are afraid. And the moment we give up hope, we give up our lives. The world we live in is disintegrating into a place of malice and hatred, where we need hope and find it harder. In this world of fear, hope to find better, but easier said than done, the more meaningful life of faith will make life meaningful.',
                 "Only when you understand the true meaning of life can you live truly. Bittersweet as life is, it's still wonderful, and it's fascinating even in tragedy. If you're just alive, try harder and try to live wonderfully.",
                 "I believe there is a person who brings sunshine into your life. That person may have enough to spread around. But if you really have to wait for someone to bring you the sun and give you a good feeling, then you may have to wait a long time.",
                 "All of us have read thrilling stories in which the hero had only a limited and specified time to live. Sometimes it was as long as a year, sometimes as short as 24 hours. But always we were interested in discovering just how the doomed hero chose to spend his last days or his last hours. I speak, of course, of free men who have a choice, not condemned criminals whose sphere of activities is strictly delimited.",
                 "Such stories set us thinking, wondering what we should do under similar circumstances. What events, what experiences, what associations should we crowd into those last hours as mortal beings, what regrets?",
                 "Sometimes I have thought it would be an excellent rule to live each day as if we should die tomorrow. Such an attitude would emphasize sharply the values of life. We should live each day with gentleness, vigor and a keenness of appreciation which are often lost when time stretches before us in the constant panorama of more days and months and years to come. There are those, of course, who would adopt the Epicurean motto of “Eat, drink, and be merry”. But most people would be chastened by the certainty of impending death.",
                 "In stories the doomed hero is usually saved at the last minute by some stroke of fortune, but almost always his sense of values is changed. He becomes more appreciative of the meaning of life and its permanent spiritual values. It has often been noted that those who live, or have lived, in the shadow of death bring a mellow sweetness to everything they do.",
                 "Most of us, however, take life for granted. We know that one day we must die, but usually we picture that day as far in the future. When we are in buoyant health, death is all but unimaginable. We seldom think of it. The days stretch out in an endless vista. So we go about our petty tasks, hardly aware of our listless attitude toward life.",
                 "The same lethargy, I am afraid, characterizes the use of all our faculties and senses. Only the deaf appreciate hearing, only the blind realize the manifold blessings that lie in sight. Particularly does this observation apply to those who have lost sight and hearing in adult life. But those who have never suffered impairment of sight or hearing seldom make the fullest use of these blessed faculties. Their eyes and ears take in all sights and sounds hazily, without concentration and with little appreciation. It is the same old story of not being grateful for what we have until we lose it, of not being conscious of health until we are ill.",
                 "I have often thought it would be a blessing if each human being were stricken blind and deaf for a few days at some time during his early adult life. Darkness would make him more appreciative of sight; silence would teach him the joys of sound.",
                 "Youth is not a time of life; it is a state of mind; it is not a matter of rosy cheeks, red lips and supple knees; it is a matter of the will, a quality of the imagination, a vigor of the emotions; it is the freshness of the deep springs of life.",
                 "Youth means a temperamental predominance of courage over timidity, of the appetite for adventure over the love of ease. This often exists in a man of 60 more than a boy of 20. Nobody grows old merely by a number of years. We grow old by deserting our ideals.",
                 "Years may wrinkle the skin, but to give up enthusiasm wrinkles the soul. Worry, fear, self-distrust bows the heart and turns the spirit back to dust.",
                 "Whether 60 or 16, there is in every human being’s heart the lure of wonders, the unfailing appetite for what’s next and the joy of the game of living. In the center of your heart and my heart, there is a wireless station; so long as it receives messages of beauty, hope, courage and power from man and from the infinite, so long as you are young.",
                 "When your aerials are down, and your spirit is covered with snows of cynicism and the ice of pessimism, then you’ve grown old, even at 20; but as long as your aerials are up, to catch waves of optimism, there’s hope you may die young at 80.",
                 "A man may usually be known by the books he reads as well as by the company he keeps; for there is a companionship of books as well as of men; and one should always live in the best company, whether it be of books or of men.",
                 "A good book may be among the best of friends. It is the same today that it always was, and it will never change. It is the most patient and cheerful of companions. It does not turn its back upon us in times of adversity or distress. It always receives us with the same kindness; amusing and instructing us in youth, and comforting and consoling us in age.",
                 "Men often discover their affinity to each other by the mutual love they have for a book just as two persons sometimes discover a friend by the admiration which both entertain for a third. There is an old proverb, ‘Love me, love my dog.” But there is more wisdom in this:” Love me, love my book.” The book is a truer and higher bond of union. Men can think, feel, and sympathize with each other through their favorite author. They live in him together, and he in them.",
                 "A good book is often the best urn of a life enshrining the best that life could think out; for the world of a man’s life is, for the most part, but the world of his thoughts. Thus the best books are treasuries of good words, the golden thoughts, which, remembered and cherished, become our constant companions and comforters.",
                 "Books possess an essence of immortality. They are by far the most lasting products of human effort. Temples and statues decay, but books survive. Time is of no account with great thoughts, which are as fresh today as when they first passed through their author’s minds, ages ago. What was then said and thought still speaks to us as vividly as ever from the printed page. The only effect of time have been to sift out the bad products; for nothing in literature can long survive e but what is really good.",
                 "Books introduce us into the best society; they bring us into the presence of the greatest minds that have ever lived. We hear what they said and did; we see the as if they were really alive; we sympathize with them, enjoy with them, grieve with them; their experience becomes ours, and we feel as if we were in a measure actors with them in the scenes which they describe.",
                 "The great and good do not die, even in this world. Embalmed in books, their spirits walk abroad. The book is a living voice. It is an intellect to which on still listens."


                ]
    #trainTest = ['How are you','Fine thank you','How do you thank me?','you are me?']

    trainGenerator = [[s.split(' ') for s in t.split('.')] for t in trainTest]
    #print(trainGenerator)

    Bondles = {} # {str:Bonded}
    embsize = 15
    lr = 5
    epoch = 3

    device = 'cpu'#'cuda' if torch.cuda.is_available() else 'cpu'


    for paragraph in trainGenerator:
        for sentence in paragraph:
            for worda in range(len(sentence)):
                #print(sentence[worda],worda)

                for wordb in range(len(sentence)):

                    if sentence[worda] == sentence[wordb]: continue
                    if sentence[worda] not in Bondles.keys():
                        Bondles[sentence[worda]] = Bundle(Vec(embnum=embsize, value=sentence[worda], device=device), lr=lr)

                    if sentence[wordb] not in Bondles.keys():
                        Bondles[sentence[wordb]] = Bundle(Vec(embnum=embsize, value=sentence[wordb], device=device), lr=lr)

                    Bondles[sentence[worda]].add(Bondles[sentence[wordb]],distance=abs(worda - wordb))


    for e in range(epoch):
        lossSum = 0
        with tqdm(range(len(Bondles)), desc=f'epoch : {e + 1} processingd') as t:
            for Bondle in Bondles.values():
                loss = Bondle.forward(Bondles)
                if not loss.isnan():
                    lossSum += loss
                else:
                    Ut.raiseError(f"Need TensorVec object to calculate difference, got {type(differences)}", sys._getframe().f_code.co_name)

                t.update(1)

            print(f'epoch : {e + 1}  loss : {lossSum / len(Bondles)}')
    '''
    for e in range(epoch):
        lossSum = 0
        count = 0
        with tqdm(range(loop), desc=f'epoch : {e+1} processingd') as t:
            for sentence in trainGenerator:
                for wordi in range(len(sentence)):
                    for wordj in range(len(sentence)):
    
                        t.update(1)
                        if sentence[wordi] == sentence[wordj]:continue
                        #print(Bonds[sentence[wordi]])
                        loss = Bonds[sentence[wordi]](Bonds[sentence[wordj]], distance=abs(wordj - wordi),optim=True)
                        # print(Bonds[sentence[wordi]])
                        # print(Bonds[sentence[wordj]])
                        # print(wordj - wordi, loss)
                        if not loss.isnan():
                            lossSum += loss.item()
                            count += 1
                            # print(sentence[wordi],sentence[wordj],loss)
                        # print(loss)
                        else:
                            print(Bonds[sentence[wordi]],Bonds[sentence[wordj]])
        print(f'epoch : {e+1}  loss : {lossSum / count}')
        time.sleep(0.01)
    '''

    for b in Bondles.values():
        print(b)


    while True:
        wordsIn = [Bondles[input('\nEnter a word : ')]]
        length=input('length:')

        print(wordsIn[0].val(), end=' ')
        for i in range(int(length)):
            n = generate(wordsIn,bondlist=Bondles)
            wordsIn.append(Bondles[n])
            print(n,end=' ')

        #for w in wordsIn:
        #    print(w.val(),end=' ')
