# Darwin Bot - Journey documentation  
GitHub Repository (View full commit history here): ++[https://github.com/navyagoyal10/darwin-bot](https://github.com/navyagoyal10/darwin-bot)++  
Overview   
This document explains the logic behind my fitness function and custom metrics for the Darwin Bot. Training this neural network required a lot of trial and error. The AI constantly tried to exploit physics glitches like falling and sliding, hopping, or dragging itself so I had to design a reward system that strictly enforced a proper walking gait.  
Body Configuration   
I initially tested a 2-legged bot, but it quickly got stuck into "hopping" . I switched to a 4-legged organism because it provides a more stable base. This made it easier for the bot to learn balance while still allowing it to achieve high speeds. I kept the leg proportions fairly balanced (thigh: 0.45m, shin: 0.38m) to allow for long strides without making the creature too tall to manage.  
Custom Metrics (my_metrics):  
 The engine runs at 60 frames per second. I wrote a custom metrics function to capture specific behaviors on a per-frame basis so I could penalize or reward them in the final score:  
1. backward_frames: Flags any frame where the bot moves backward faster than 2 px/s. I added this when the bot figured out walking backward was easier.  
2. fwd_speed: Captures only positive horizontal velocity to encourage forward momentum.  
3. air_frames: Disregards frames where zero feet are touching the ground. This was crucial for detecting and stopping the jumping/hopping exploit.  
4. angles: Tracks frames where the torso is tilted, helping me measure and enforce better posture.  
The Fitness Function (my_fitness)   
 The algorithm's only goal is to maximize this score.  
1. The Hard Constraint (Falling) If the bot's torso hits the ground or tilts too far, I apply a massive penalty (-10.0) combined with a tiny fraction of the distance (* 0.1). This guarantees that a fallen bot always scores lower than a walking one, which completely prevents the bot from learning the "fall-and-slide" exploit.  
2. Distance Distance is the primary objective. I multiplied it by 39.0 so the neural network is strongly forced to cover as much ground as possible.  
3. Form and Posture  To force a proper walking gait instead of dragging or hopping, I added rewards for taking actual steps (step_count) and using multiple legs (legs_active). I also added penalties for tilted posture and spending time completely in the air.  
Why I Chose These Specific Weights  
Finding the right multipliers was the hardest part of the project. If a secondary reward (like smoothness) outweighs the primary goal (distance), the AI will just stand perfectly still. I chose these weights based on their relative magnitudes to create a balanced reward system:  
1. Distance @ 39.0: This had to be the heaviest weight by a significant margin. No matter what other penalties apply, the bot's overarching drive must always be to cover ground.  
2. Smoothness (14.0) and the backward walking penalty (10.0). I initially tried squaring the smoothness score, but it overpowered the distance metric. A multiplier of 14.0 was a good balance , high enough to stop the hopping exploit, but low enough that the bot wouldn't sacrifice distance to achieve it. The 10.0 backward penalty acted as a strict one.  
3. Step Count (5.0) and Legs Active (7.0) are moderate rewards. They are large enough to guide the bot toward a 4-legged gait rather than dragging itself on two legs, but small enough that a clumsy bot taking lots of steps won't outscore a smooth, fast bot.  
4. Angles (4.5) and Air Frames (2.6) are penalties which ensure a better gait. They gently push the bot toward upright, grounded movement . The Forward Speed (0.9) acts as a means to encourage velocity.  
Through this balancing act, the bot finally broke the 30-meter mark. I ultimately settled on this configuration because it prioritizes a slightly more aesthetic, stable walk (averaging ~26 meters) over the chaotic, barely-balanced sprint of the 30-meter anomaly.  
  
  
- [ ] My OWN log while I was doing the process:  
Ok so here is some more learnings of mine.  
The fitness function I design needs to reward good movements and penalise bad movements,  
I can also define my own metrics and think about the most effective movement strategies.  
1. Test Run-1 Ok so I modified the fitness function by giving a penalty for torso tilt, since that makes it prone to falling and deteriorates walk quality. What I did : 2-legged bot got stuck in a "hopping local optimum" so I shifted to a 4-legged topology to enforce a cleaner gait. Observations:  
    1. It is going quite okay , I mean the distance= 5.66 m and is not falling quickly  
    2. The gait is not good, it is basically hopping using both legs instead of a normal walk, maybe its easier for it to accelerate that way, so I’ll try changing the score with respect to smoothness since hopping alters smoothness  
    3. I’m also gonna go for a 4 legged creature now, since it’s more difficult to hop as a 4 legged thing plus balance is easier and distance covered is more)  
2. Test Run-2  
	I changed the creature to a 4 legged one(easier balance so easier to learn) 	again it started hopping even though I gave more weight to torso angle, so now im gonna try to give more weight to smoothness maybe , so that it knows not to do it, so maybe ill multiply the score with smoothness(message from the future: I ended up realising multiplying the smoothness was reducing the weight of the distance, so I altered the smoothness to be additive at the end)  
3.  Test Run-3  
	ok this worked, the multiplying thing, definitely getting better fitness scores and distances   
	but now the curve is kinda getting static, in the sense ki its not improving much stuck on 13 m from gen 3 to gen 50  
	So now imma work the config.ini file to change the compatibility threshold because I need to see that improvement and the fitness threshold because the scores are too high.  
4. Test Run-4  
	ok so now the creature has hit the 18 m mark but it is only using 2 legs to move instead of 4, it has curled up the other 2 legs to reduce friction so the walking quality is pathetic, what to do now? Smoothness won’t work, probably need to do smth with steps  
(Increased the step count multiplier to give more weight to it)  
5. Test Run-5  
Hit 18.48 m! That too with it using 3 legs, fairly a good one! But now ive gotta adjust the smoothness ka weight to maintain the symmetry(Tried squaring )  
6.  Test-6  
  So now distance Is down to 10.27 but squaring has given too much weight to smoothness(more thaan step count, so Imma reduce the multiplier )  
7.  Test-7  
Ok so I added the feet down parameter to ensure all 4 legs were touching and gave it same weight as step count, but function has started to go backwards instead of forwards so now imma add the forward velocity thing(to discourage backward walking) and add a penalty for backward walking!  
8. Test-8  
Added a new rubric for forward velocity and must I say, it has worked spectacularly  
NEW MAXIMA UNLOCKED! - 21.32 m  
Also a beautiful fitness curve (if I may add)  
  
Was getting a plateau curve so reduced the compatibility threshold to 1.5   
Was getting a super high fit score so reduced the weights of everything  
I’ve just been experimenting with weights of air frames and step counts and forward speed for a while now  
Im gonna try experimenting with thigh length now, since longer thighs mean longer steps and thus more velocity  
The stride is better now, so ill just give it the forward speed push, gonna risk this, take the multiplier to 3  
Also the creature is not swinging its legs that well after the increased thigh length so ill increase the hip range to 1 or smth   
(Ok so after this point the creature was losing its balance and honestly I was getting reduced distances with the higher thigh length and all , so I reverted back to the original specs)  
  
I have also tried adding more metrics, and experimented but none of that seemed to workout that well so I have attached all the different walks I achieved and the best results.  
After trying like a thousand times to find the perfect ratio for the weights of the rubrics, finally I’ve reached a major breakthrough   
This beautiful code has reached 30.85 m !!! (200 gens but it hit 30.85 on the 65th gen)(gotta attach proof)  
  
Love it, although the walk has slightly worsened , but hey a new maxima is a new win!  
  
  
Ok I’ve tweaked the code a little bit more to give it a better walk, I feel like an almost 3 legged walk(mostly 2) with a good distance (hitting around 26 m) is pretty decent.  
  
NOTE: This whole documentation gives an idea of the mental turmoil and small instances of victories that I’ve gone through for the past week (I might not have all these test codes with me but I have some screenshots and final tweaked version of the code)(everything is on my GitHub)  
  
GitHub repo link : [https://github.com/navyagoyal10/darwin-bot](https://github.com/navyagoyal10/darwin-bot)  
  
