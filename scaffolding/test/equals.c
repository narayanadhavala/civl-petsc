#include <pointer.cvh>

$input int N;
int len = 3;
$assume(N > 0 & len > N);
$input double A_Real[N];

int main(void) {
  double values[N], values2[N];
  for (int i = 0; i < N; i++) {
    values[i] = A_Real[i];
    values2[i] = A_Real[i];
    $print("values  = ", values[i], "\n", "values2 = ", values2[i], "\n\n");
  }
  //$assert($equals(&values, &values2));
}
