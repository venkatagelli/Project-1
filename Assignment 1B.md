#### Kernels

Whenever we look at any image, we see lots of distinguishable features. We need a mechanism to extract those features. We use Kernels for this purpose. Kernels are also called Feature Extractors, Filters, 3x3 Matrix. These filters are used to extract features like Vertical Edges, Horizontal Edges, Gradients, Textures, Patterns, Parts of Objects, Objects etc.

Mostly we use odd number kernels. Even number kernels do not have the concept of middle. Hence we can't talk about the things on right and left. It is very difficult to create anything symmetric using even kernels.

1x1 kernel is mostly used to increase the number of channels.

#### Channels

Channels are thought of similar-feature-bags. One image is divided into as many features as possible and these features are combined to be called channels. Complex things are made from small features. More features divided into multiple channels. For example, if there are 50 distinguishable features then there will be 50 channels.



#### Why should we only (well mostly) use 3x3 Kernels?

3x3 Kernels are highly optimized.

3x3 Kernels add fewer number of parameters i.e 9 when compared with higher order Kernels. 

Two 3x3 kernels will add 18 parameters to the network and is equivalent to one 5x5 kernel where as A 5x5 Kernel will add 25 parameters. 



#### How many times do we need to perform 3x3 convolution operation to reach 1x1 from 199x199 (show calculations)

100 times we need to perform 3x3 convolution operation to reach 

A 3x3 convolution reduces 2x2 from input. i.e a 3x3 convolution applied on 5x5 reduces it to 3x3.

Notation : ''|'' represents 3x3 layer and 199 represents a matrix of 199x199, 197 represents a matrix of 197x197 so on ...

i = 199
for j in range (1,101):
     print(str(i)+' |');
     i = i-2

#### Output:

199 |
197 |
195 |
193 |
191 |
189 |
187 |
185 |
183 |
181 |
179 |
177 |
175 |
173 |
171 |
169 |
167 |
165 |
163 |
161 |
159 |
157 |
155 |
153 |
151 |
149 |
147 |
145 |
143 |
141 |
139 |
137 |
135 |
133 |
131 |
129 |
127 |
125 |
123 |
121 |
119 |
117 |
115 |
113 |
111 |
109 |
107 |
105 |
103 |
101 |
99 |
97 |
95 |
93 |
91 |
89 |
87 |
85 |
83 |
81 |
79 |
77 |
75 |
73 |
71 |
69 |
67 |
65 |
63 |
61 |
59 |
57 |
55 |
53 |
51 |
49 |
47 |
45 |
43 |
41 |
39 |
37 |
35 |
33 |
31 |
29 |
27 |
25 |
23 |
21 |
19 |
17 |
15 |
13 |
11 |
9 |
7 |
5 |
3 |
1