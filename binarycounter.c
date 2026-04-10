#include <stdio.h>
#include <string.h>

int main() {
    char binary[100];
    int state = 0; // q0 = even (ACCEPT), q1 = odd

    printf("Enter binary string: ");
    scanf("%s", binary);

    printf("\n--- Execution Trace ---\n");

    for(int i = 0; i < strlen(binary); i++) {
        char ch = binary[i];

        printf("Input: %c | Current State: q%d\n", ch, state);

        if(ch == '1') {
            if(state == 0)
                state = 1;
            else
                state = 0;
        }
        else if(ch != '0') {
            printf("Invalid input detected!\n");
            return 0;
        }
    }

    printf("\n--- Final Result ---\n");

    if(state == 0)
        printf("ACCEPTED (Even number of 1's)\n");
    else
        printf("REJECTED (Odd number of 1's)\n");

    return 0;
}