% Perception
nn(perc_net_version_3_NeSy_danger_pedestrian,[Img,Speed],X,[0,1,2,3]) :: cell_danger_pedestrian(Img,Speed,X).
nn(perc_net_version_3_NeSy_speed_zone,[MNIST],SZ,[0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0]) :: cell_speed_zone(MNIST,SZ).
nn(perc_net_version_3_NeSy_traffic_light,[Img],X,[green, orange, red]) :: cell_traffic_light(Img,X).
nn(perc_net_version_3_NeSy_danger,[Img,Speed,Yb,Yi],X,[0,2,3]) :: cell_danger(Img,Speed,Yb,Yi,X).


%%%%    TRAFFIC RULES    %%%%
% Rules - pedestrian
ped_brake(Img,Speed) :-
    cell_danger_pedestrian(Img,Speed,D),
    D = 3.
ped_idle(Img,Speed) :-
    cell_danger_pedestrian(Img,Speed,D),
    D = 2.
ped_follow(Img,Speed) :-
    cell_danger_pedestrian(Img,Speed,D),
    D = 1.
ped_acc(Img,Speed) :-
    cell_danger_pedestrian(Img,Speed,D),
    D = 0.

% Rules - speed zone
speed_zone_brake(MNIST,Speed) :-
    cell_speed_zone(MNIST,SZ),
    Speed > SZ.

speed_zone_follow(MNIST,Speed) :-
    cell_speed_zone(MNIST,SZ),
    round(Speed) == SZ.

speed_zone_accelerate(MNIST,Speed) :-
    cell_speed_zone(MNIST,SZ),
    round(Speed) < SZ.


% Rules - distance
brake_dist(Y,Speed) :-
    Acc = 0.1,
    Frames_brake = Speed/(2*Acc),
    Y is (-(2*Acc)*Frames_brake**2)/2 + Speed*Frames_brake.

idle_dist(Y,Speed) :-
    Acc = 0.1,
    Frames_idle = Speed/(Acc/2),
    Y is (-(Acc/2) * Frames_idle ** 2) / 2 + Speed * Frames_idle.


% Rules - traffic light
traffic_light_brake(Img,MNIST,Speed) :-
    cell_traffic_light(Img,C),
    brake_dist(Yb,Speed),
    idle_dist(Yi,Speed),
    cell_danger(Img,Speed,Yb,Yi,D),
    (C = red ; C = orange),
    D = 3,
    Speed > 0.2.

traffic_light_idle(Img,MNIST,Speed) :-
    cell_traffic_light(Img,C),
    brake_dist(Yb,Speed),
    idle_dist(Yi,Speed),
    cell_danger(Img,Speed,Yb,Yi,D),
    ((D = 3, (C = red ; C = orange), Speed =< 0.2);
    (D = 2, (C = orange))).

traffic_light_accelerate(Img,MNIST,Speed) :-
    cell_traffic_light(Img,C),
    brake_dist(Yb,Speed),
    idle_dist(Yi,Speed),
    cell_danger(Img,Speed,Yb,Yi,D),
    (C = green; D = 0).


% Control
% 0 --> accelerate
% 1 --> break
% 2 --> idle
% 3 --> keep pace

action(Img,MNIST,Speed,1) :-
    speed_zone_brake(MNIST,Speed);
    traffic_light_brake(Img,MNIST,Speed);
    ped_brake(Img,Speed).
action(Img,MNIST,Speed,2) :-
    (traffic_light_idle(Img,MNIST,Speed);
    ped_idle(Img,Speed)),
    \+ action(Img,MNIST,Speed,1).
action(Img,MNIST,Speed,3) :-
    (speed_zone_follow(MNIST,Speed);
    ped_follow(Img,Speed)),
    \+ action(Img,MNIST,Speed,1),
    \+ action(Img,MNIST,Speed,2).
action(Img,MNIST,Speed,0) :-
    (traffic_light_accelerate(Img,MNIST,Speed);
    ped_acc(Img,Speed)),
    \+ action(Img,MNIST,Speed,1),
    \+ action(Img,MNIST,Speed,2),
    \+ action(Img,MNIST,Speed,3).
