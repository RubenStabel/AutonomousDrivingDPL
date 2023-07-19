% Perception
nn(perc_net_version_2_NeSy_ped,[Img,Speed],X,[0,1,2,3]) :: cell_ped(Img,Speed,X).
nn(perc_net_version_2_NeSy_speed_zone,[MNIST],SZ,[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0]) :: cell_spd(MNIST,SZ).


% Rules
speed_zone_brake(SZ,S) :-
    S > SZ.

speed_zone_follow(SZ,S) :-
    round(S) == SZ.


% Control
% 0 --> accelerate
% 1 --> break
% 2 --> idle
% 3 --> keep pace

action(Img,MNIST,Speed,1) :- cell_ped(Img,Speed,3);(cell_spd(MNIST,SZ1), speed_zone_brake(SZ1,Speed)).
action(Img,MNIST,Speed,2) :- cell_ped(Img,Speed,2), \+ action(Img,MNIST,Speed,1).
action(Img,MNIST,Speed,3) :- (cell_ped(Img,Speed,1);(cell_spd(MNIST,SZ3), speed_zone_follow(S3,Speed))), \+ action(Img,MNIST,Speed,1), \+ action(Img,MNIST,Speed,2).
action(Img,MNIST,Speed,0) :- cell_ped(Img,Speed,0),\+ action(Img,MNIST,Speed,1), \+ action(Img,MNIST,Speed,2), \+ action(Img,MNIST,Speed,3).
