data { real a[5]; } model { for (n in a[1]:5) a[n] <- n; }