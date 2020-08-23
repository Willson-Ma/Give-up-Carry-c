turtles-own
[
  data-obtained?
  data-spy?

]

globals [ i mylist mzx j lst sublst subsublst]

to setup
  clear-all
  setup-nodes
  setup-network
  ask n-of initial-outbreak-size turtles
    [
      source-data
      set label "ID:0 Data:abced"
      set label-color red
      set mylist ["a" "b" "c" "d" "e"]
      set mzx who

    ]
  ask links [ set color 67 ]
  reset-ticks
end

to setup-nodes
  set-default-shape turtles "circle"
;  set i 1
  while [count turtles < number-of-nodes]
  [ifelse count turtles < number-of-nodes - 1
    [create-turtles 1
    [
      set size 2
      ; for visual reasons, we don't put any nodes *too* close to the edges
      setxy (random-xcor * 0.95) (random-ycor * 0.95)

      set label who
      set label-color red


      agent-normal

    ]
    ]

    [
      create-turtles 1
      [
        set size 2
        setxy (random-xcor * 0.95) (random-ycor * 0.95)

        set label "spy"
        set label-color red
        agent-spy
      ]
    ]
  ]
end

to setup-network
  let num-links (average-node-degree * number-of-nodes) / 2
  while [count links < num-links ]
  [
    ask one-of turtles
    [
      let choice (min-one-of (other turtles with [not link-neighbor? myself])
                   [distance myself])
      if choice != nobody [ create-link-with choice ]
    ]
  ]

  repeat 10
  [
    layout-spring turtles links 0.3 (world-width / (sqrt number-of-nodes)) 1
  ]
end

to go
  if all? turtles [data-obtained?]
    [ stop ]

  spread-data
;  respond-data

  tick
end

to source-data
  set data-obtained? true
  set data-spy? false
  set color blue

end

to agent-normal
  set data-obtained? false
  set data-spy? false
  set color white
end

to agent-spy
  set data-obtained? false
  set data-spy? true
  set color yellow
  ask my-links [ set color gray - 2 ]
end

to spread-data
  ifelse add-limit?
  [
    ifelse ticks < 1
    [
    ask turtles with [data-obtained?]
      [ ifelse any? link-neighbors with [data-spy?]
        [
          ask link-neighbors with [not data-spy?]
          [ if random-float 100 < limit-tolerance
            [ source-data ]
;        let random-list list n-of number-of-data-received mylist
;        ask n-of number-of-data-received mylist [set list-mzx]


           set lst n-of number-of-data-received mylist
           set label lst
           set label-color red

          ]

          ask turtles with [data-spy?]
          [
            if random-float 100 < protection-tolerance
            [source-data]
            let spydata random 6
            set color orange
            let spylabel n-of spydata mylist
            set label spylabel
            set label-color red
          ]
        ]


        [
        ask link-neighbors
        [ if random-float 100 < limit-tolerance
          [ source-data ]


           set lst n-of number-of-data-received mylist
           set label lst
           set label-color red


        ]
        ]
    ]
    ]

    [
      ask turtles with [data-obtained?]
      [
        ask link-neighbors with [not data-obtained?]
          [ if random-float 100 < limit-tolerance
            [ source-data ]


           set lst n-of number-of-data-received mylist
           set label lst
           set label-color red
          ]
      ]

      ask turtles with [data-spy?]
          [if any? link-neighbors with [data-obtained?]
            [
            source-data
            set color orange
            let hahaha random 5
            set lst n-of hahaha mylist
            set label lst
            set label-color red
            ]
          ]
    ]
  ]

 [
        ask turtles with [data-obtained?]
        [ ask link-neighbors
        [ if random-float 100 < limit-tolerance
        [ source-data ]
        set color blue
        set label "a,b,c,d,e"
        set label-color red
        ]
        ]



  ]




end




to-report count-data-spy report count turtles with [ data-spy? ]
end
to-report count-data-obtained report count turtles with [ data-obtained? ]
end
to-report count-susceptible report count turtles with [not data-obtained? and not data-spy?]
end
@#$#@#$#@
GRAPHICS-WINDOW
729
12
1391
675
-1
-1
15.95122
1
10
1
1
1
0
0
0
1
-20
20
-20
20
1
1
1
ticks
30.0

SLIDER
168
221
511
254
number-of-data-received
number-of-data-received
1
5
2.0
1
1
NIL
HORIZONTAL

BUTTON
220
128
315
168
NIL
setup
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
330
128
425
168
NIL
go
T
1
T
OBSERVER
NIL
NIL
NIL
NIL
0

PLOT
129
363
714
664
Network Status
time
% of nodes
0.0
52.0
0.0
100.0
true
true
"" ""
PENS
"Nodes that have obtained data" 1.0 0 -13345367 true "" "plot (count turtles with [data-obtained?]) / (count turtles) * 100"
"Spy nodes that don't have obtained data" 1.0 0 -4079321 true "" "plot (count turtles with [data-spy?]) / (count turtles) * 100"
"Spy nodes that have obtained data" 1.0 0 -955883 true "" "plot (count turtles with [color = orange]) / (count turtles) * 100"

SLIDER
161
17
513
50
number-of-nodes
number-of-nodes
1
30
10.0
1
1
NIL
HORIZONTAL

SLIDER
163
87
512
120
initial-outbreak-size
initial-outbreak-size
1
5
1.0
1
1
NIL
HORIZONTAL

SLIDER
162
52
513
85
average-node-degree
average-node-degree
1
number-of-nodes - 1
4.0
1
1
NIL
HORIZONTAL

SWITCH
265
177
392
210
add-limit?
add-limit?
0
1
-1000

SLIDER
167
270
512
303
limit-tolerance
limit-tolerance
0
10
2.0
1
1
NIL
HORIZONTAL

SLIDER
167
316
512
349
protection-tolerance
protection-tolerance
0
20
2.0
1
1
NIL
HORIZONTAL

@#$#@#$#@
## What is it?

This model demonstrates the spread of open source data through the network. Although the model is somewhat abstract, one explanation is that each node represents an agent, and we are modeling the propagation process of open source data (or information) through this network. Each node may be in one of three states: data source, agent waiting to receive data, or spy (agent that maliciously steals data). In the academic literature, this model is similar to the SIR model.

## How does this work
The model is divided into two versions, limited and unrestricted versions, you can choose by yourself, the unrestricted version each node has 100% data access, because the data is completely open source without any verification and protection measures, so data dissemination Very fast and extremely large data transfer volume. The following is a detailed description of the model version with increased restrictions.

At each time step (scale), each node that has obtained data (blue, including the source data node) will try to transmit the data to all its authorized neighbors. Each neighbor node that is susceptible to infection will be set by the limit-tolerance slider to obtain the data probability, and the maximum value cannot exceed the limit-tolerance value. That is, every time go is executed, each node that has not obtained data will randomly be a number of 1-100. When this value is less than the limit-tolerance rate, it indicates that this node can obtain data, and after obtaining the data, it will become Source the same color. This may correspond to the possibility that an agent on the open source data collaboration system actually obtains data and collaborates.

Spy nodes (yellow) represent spies who want to steal data illegally. If the spy node is directly connected to the source data node, then we assume that it cannot obtain data from other neighbor nodes because the spy wants to reduce the possibility of their exposure. This means that spy nodes have a chance to obtain all the data without reconstruction, through illegal means. The probability of successful stealing is affected by the protection-tolerance rate of the source data node. The same as the above operation, the spy node will have a number of 1-100 each time it goes. When this value is less than the protection-tolerance rate, it means that it has passed the protection set by the data source and has stolen random amount of data (possibly all), the node will turn orange to indicate this behavior. This may correspond to the anti-virus software and security patches set by the actual open source database. These protective measures can prevent the data source from being stolen. When the spy node is not directly connected to the data source, it can only try to obtain data through its neighbor nodes, but this means that it cannot obtain all the data because its neighbor nodes do not get all the data. Therefore, the spy node may obtain more data than neighbor nodes or less than neighbor nodes, but it must not exceed the data volume of the data source. This is also a difference from the situation where the spy node is directly connected to the data source. After obtaining the data, the spy node will disguise itself, that is, become the same color as the ordinary data node, making it difficult to detect.

## How to use it

Using the slider, set NUMBER-OF-NODES (total number of nodes) and AVERAGE-NODE-DEGREE (average number of links from each node) to the values ​​you want.

The shared collaboration network created is based on the proximity (Euclidean distance) between nodes. Randomly select a node and connect it to the nearest node that is not yet connected. Repeat this process until the network has the correct number of links to give the specified average node degree.

The INITIAL-OUTBREAK-SIZE slider determines how many nodes will be defined as data source nodes.

Then press SETUP to create the network. Press GO to run the model. Once all nodes have obtained the data, the model will stop running.

The "add-limit?", "number-of-data-received", "limit-tolerance" and "protection-tolerance" sliders can be adjusted before pressing GO or while the model is running (in the "How to work" above Discussed).

The line chart graph shows the number of nodes that obtain data in the model and the number of spy nodes (whether it is disguised) over time.

## Precautions

At the end of the run, how long did it take for all nodes to obtain all data? How does changing the AVERAGE-NODE-DEGREE and AVERAGE-NODE-DEGREE of the network affect this?

## Improving features

In the case of a limited version and the amount of data obtained is not 5, the data obtained by each node is random, that is, the response to different data is different. Can the number of nodes that obtain a certain specific data change over time in a line chart? Are the results of the model the same every time?

## Related models

Viruses on the Internet, gossip, spread.

## NETLOGO FEATURES

Links are used for modeling the network.  The `layout-spring` primitive is used to position the nodes and links such that the structure of the network is visually clear.

Though it is not used in this model, there exists a network extension for NetLogo that you can download at: https://github.com/NetLogo/NW-Extension.

## COPYRIGHT 

Copyright 2020 Zixiang Ma.
@#$#@#$#@
default
true
0
Polygon -7500403 true true 150 5 40 250 150 205 260 250

airplane
true
0
Polygon -7500403 true true 150 0 135 15 120 60 120 105 15 165 15 195 120 180 135 240 105 270 120 285 150 270 180 285 210 270 165 240 180 180 285 195 285 165 180 105 180 60 165 15

arrow
true
0
Polygon -7500403 true true 150 0 0 150 105 150 105 293 195 293 195 150 300 150

box
false
0
Polygon -7500403 true true 150 285 285 225 285 75 150 135
Polygon -7500403 true true 150 135 15 75 150 15 285 75
Polygon -7500403 true true 15 75 15 225 150 285 150 135
Line -16777216 false 150 285 150 135
Line -16777216 false 150 135 15 75
Line -16777216 false 150 135 285 75

bug
true
0
Circle -7500403 true true 96 182 108
Circle -7500403 true true 110 127 80
Circle -7500403 true true 110 75 80
Line -7500403 true 150 100 80 30
Line -7500403 true 150 100 220 30

butterfly
true
0
Polygon -7500403 true true 150 165 209 199 225 225 225 255 195 270 165 255 150 240
Polygon -7500403 true true 150 165 89 198 75 225 75 255 105 270 135 255 150 240
Polygon -7500403 true true 139 148 100 105 55 90 25 90 10 105 10 135 25 180 40 195 85 194 139 163
Polygon -7500403 true true 162 150 200 105 245 90 275 90 290 105 290 135 275 180 260 195 215 195 162 165
Polygon -16777216 true false 150 255 135 225 120 150 135 120 150 105 165 120 180 150 165 225
Circle -16777216 true false 135 90 30
Line -16777216 false 150 105 195 60
Line -16777216 false 150 105 105 60

car
false
0
Polygon -7500403 true true 300 180 279 164 261 144 240 135 226 132 213 106 203 84 185 63 159 50 135 50 75 60 0 150 0 165 0 225 300 225 300 180
Circle -16777216 true false 180 180 90
Circle -16777216 true false 30 180 90
Polygon -16777216 true false 162 80 132 78 134 135 209 135 194 105 189 96 180 89
Circle -7500403 true true 47 195 58
Circle -7500403 true true 195 195 58

circle
false
0
Circle -7500403 true true 0 0 300

circle 2
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240

cow
false
0
Polygon -7500403 true true 200 193 197 249 179 249 177 196 166 187 140 189 93 191 78 179 72 211 49 209 48 181 37 149 25 120 25 89 45 72 103 84 179 75 198 76 252 64 272 81 293 103 285 121 255 121 242 118 224 167
Polygon -7500403 true true 73 210 86 251 62 249 48 208
Polygon -7500403 true true 25 114 16 195 9 204 23 213 25 200 39 123

cylinder
false
0
Circle -7500403 true true 0 0 300

dot
false
0
Circle -7500403 true true 90 90 120

face happy
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 255 90 239 62 213 47 191 67 179 90 203 109 218 150 225 192 218 210 203 227 181 251 194 236 217 212 240

face neutral
false
0
Circle -7500403 true true 8 7 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Rectangle -16777216 true false 60 195 240 225

face sad
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 168 90 184 62 210 47 232 67 244 90 220 109 205 150 198 192 205 210 220 227 242 251 229 236 206 212 183

fish
false
0
Polygon -1 true false 44 131 21 87 15 86 0 120 15 150 0 180 13 214 20 212 45 166
Polygon -1 true false 135 195 119 235 95 218 76 210 46 204 60 165
Polygon -1 true false 75 45 83 77 71 103 86 114 166 78 135 60
Polygon -7500403 true true 30 136 151 77 226 81 280 119 292 146 292 160 287 170 270 195 195 210 151 212 30 166
Circle -16777216 true false 215 106 30

flag
false
0
Rectangle -7500403 true true 60 15 75 300
Polygon -7500403 true true 90 150 270 90 90 30
Line -7500403 true 75 135 90 135
Line -7500403 true 75 45 90 45

flower
false
0
Polygon -10899396 true false 135 120 165 165 180 210 180 240 150 300 165 300 195 240 195 195 165 135
Circle -7500403 true true 85 132 38
Circle -7500403 true true 130 147 38
Circle -7500403 true true 192 85 38
Circle -7500403 true true 85 40 38
Circle -7500403 true true 177 40 38
Circle -7500403 true true 177 132 38
Circle -7500403 true true 70 85 38
Circle -7500403 true true 130 25 38
Circle -7500403 true true 96 51 108
Circle -16777216 true false 113 68 74
Polygon -10899396 true false 189 233 219 188 249 173 279 188 234 218
Polygon -10899396 true false 180 255 150 210 105 210 75 240 135 240

house
false
0
Rectangle -7500403 true true 45 120 255 285
Rectangle -16777216 true false 120 210 180 285
Polygon -7500403 true true 15 120 150 15 285 120
Line -16777216 false 30 120 270 120

leaf
false
0
Polygon -7500403 true true 150 210 135 195 120 210 60 210 30 195 60 180 60 165 15 135 30 120 15 105 40 104 45 90 60 90 90 105 105 120 120 120 105 60 120 60 135 30 150 15 165 30 180 60 195 60 180 120 195 120 210 105 240 90 255 90 263 104 285 105 270 120 285 135 240 165 240 180 270 195 240 210 180 210 165 195
Polygon -7500403 true true 135 195 135 240 120 255 105 255 105 285 135 285 165 240 165 195

line
true
0
Line -7500403 true 150 0 150 300

line half
true
0
Line -7500403 true 150 0 150 150

pentagon
false
0
Polygon -7500403 true true 150 15 15 120 60 285 240 285 285 120

person
false
0
Circle -7500403 true true 110 5 80
Polygon -7500403 true true 105 90 120 195 90 285 105 300 135 300 150 225 165 300 195 300 210 285 180 195 195 90
Rectangle -7500403 true true 127 79 172 94
Polygon -7500403 true true 195 90 240 150 225 180 165 105
Polygon -7500403 true true 105 90 60 150 75 180 135 105

plant
false
0
Rectangle -7500403 true true 135 90 165 300
Polygon -7500403 true true 135 255 90 210 45 195 75 255 135 285
Polygon -7500403 true true 165 255 210 210 255 195 225 255 165 285
Polygon -7500403 true true 135 180 90 135 45 120 75 180 135 210
Polygon -7500403 true true 165 180 165 210 225 180 255 120 210 135
Polygon -7500403 true true 135 105 90 60 45 45 75 105 135 135
Polygon -7500403 true true 165 105 165 135 225 105 255 45 210 60
Polygon -7500403 true true 135 90 120 45 150 15 180 45 165 90

square
false
0
Rectangle -7500403 true true 30 30 270 270

square 2
false
0
Rectangle -7500403 true true 30 30 270 270
Rectangle -16777216 true false 60 60 240 240

star
false
0
Polygon -7500403 true true 151 1 185 108 298 108 207 175 242 282 151 216 59 282 94 175 3 108 116 108

target
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240
Circle -7500403 true true 60 60 180
Circle -16777216 true false 90 90 120
Circle -7500403 true true 120 120 60

tree
false
0
Circle -7500403 true true 118 3 94
Rectangle -6459832 true false 120 195 180 300
Circle -7500403 true true 65 21 108
Circle -7500403 true true 116 41 127
Circle -7500403 true true 45 90 120
Circle -7500403 true true 104 74 152

triangle
false
0
Polygon -7500403 true true 150 30 15 255 285 255

triangle 2
false
0
Polygon -7500403 true true 150 30 15 255 285 255
Polygon -16777216 true false 151 99 225 223 75 224

truck
false
0
Rectangle -7500403 true true 4 45 195 187
Polygon -7500403 true true 296 193 296 150 259 134 244 104 208 104 207 194
Rectangle -1 true false 195 60 195 105
Polygon -16777216 true false 238 112 252 141 219 141 218 112
Circle -16777216 true false 234 174 42
Rectangle -7500403 true true 181 185 214 194
Circle -16777216 true false 144 174 42
Circle -16777216 true false 24 174 42
Circle -7500403 false true 24 174 42
Circle -7500403 false true 144 174 42
Circle -7500403 false true 234 174 42

turtle
true
0
Polygon -10899396 true false 215 204 240 233 246 254 228 266 215 252 193 210
Polygon -10899396 true false 195 90 225 75 245 75 260 89 269 108 261 124 240 105 225 105 210 105
Polygon -10899396 true false 105 90 75 75 55 75 40 89 31 108 39 124 60 105 75 105 90 105
Polygon -10899396 true false 132 85 134 64 107 51 108 17 150 2 192 18 192 52 169 65 172 87
Polygon -10899396 true false 85 204 60 233 54 254 72 266 85 252 107 210
Polygon -7500403 true true 119 75 179 75 209 101 224 135 220 225 175 261 128 261 81 224 74 135 88 99

wheel
false
0
Circle -7500403 true true 3 3 294
Circle -16777216 true false 30 30 240
Line -7500403 true 150 285 150 15
Line -7500403 true 15 150 285 150
Circle -7500403 true true 120 120 60
Line -7500403 true 216 40 79 269
Line -7500403 true 40 84 269 221
Line -7500403 true 40 216 269 79
Line -7500403 true 84 40 221 269

x
false
0
Polygon -7500403 true true 270 75 225 30 30 225 75 270
Polygon -7500403 true true 30 75 75 30 270 225 225 270
@#$#@#$#@
NetLogo 6.1.1
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
default
0.0
-0.2 0 0.0 1.0
0.0 1 1.0 0.0
0.2 0 0.0 1.0
link direction
true
0
Line -7500403 true 150 150 90 180
Line -7500403 true 150 150 210 180
@#$#@#$#@
0
@#$#@#$#@
