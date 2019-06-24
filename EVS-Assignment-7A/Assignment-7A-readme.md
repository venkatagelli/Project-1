Channel size calculation formula

![1561379630691](C:\Users\VENKATA\AppData\Roaming\Typora\typora-user-images\1561379630691.png)

Receptive Field size calculation formula

![1561379677870](C:\Users\VENKATA\AppData\Roaming\Typora\typora-user-images\1561379677870.png)



J-out ==> Jump calculated for next  operation

J-in ==> Jump of the current operation

r-in ==> incoming receptive field size

r-out ==> outgoing receptive field size





| Operation                 | Input | output | Receptive field | jump |
| ------------------------- | ----- | ------ | --------------- | ---- |
| Input                     | 112   | 112    | 1               | 1    |
| Convolution k=7, s=2, p=0 | 112   | 53     | 7               | 1    |
| Maxpooling k=3, s=2       | 53    | 26     | 11              | 2    |
| Convolution k=3, s=1      | 26    | 24     | 19              | 4    |
| Maxpooling k=3, s=2       | 24    | 12     | 27              | 4    |
| Inception 3a, k=1         | 12    | 12     | 27              | 8    |
| Inception 3b, k=1         | 12    | 12     | 27              | 8    |
| Maxpooling k=3, s=2       | 12    | 6      | 43              | 8    |
| Inception 4a, k=1         | 6     | 6      | 43              | 16   |
| Inception 4b, k=1         | 6     | 6      | 43              | 16   |
| Inception 4c, k=1         | 6     | 6      | 43              | 16   |
| Inception 4d, k=1         | 6     | 6      | 43              | 16   |
| Inception 4e, k=1         | 6     | 6      | 43              | 16   |
| Maxpooling k=3, s=2       | 6     | 3      | 75              | 16   |
| Inception 5a, k=1         | 6     | 3      | 75              | 32   |
| Inception 5b, k=1         | 6     | 3      | 75              | 32   |
| Average pool              | 3     | 0      | 267             | 32   |