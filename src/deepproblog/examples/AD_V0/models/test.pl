nn(ad_baseline_net,[Img],X,[high,medium,low]) :: danger_level(Img,X).

autonomous_driving_baseline(Img,X) :- action(Img,X).

action(Img,0) :-  danger_level(Img,low).
action(Img,1) :- danger_level(Img,high).
action(Img,2)  :-  danger_level(Img,medium).
