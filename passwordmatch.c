#include <stdio.h>
#include <ctype.h>
#include <string.h>

int main() {
    char password[100];
    int hasLower = 0, hasUpper = 0, hasDigit = 0;

    printf("Enter password: ");
    scanf("%s", password);

    printf("\n--- Execution Trace ---\n");

    for(int i = 0; i < strlen(password); i++) {
        char ch = password[i];

        if(islower(ch)) {
            hasLower = 1;
            printf("Input: %c -> Lowercase\n", ch);
        }
        else if(isupper(ch)) {
            hasUpper = 1;
            printf("Input: %c -> Uppercase\n", ch);
        }
        else if(isdigit(ch)) {
            hasDigit = 1;
            printf("Input: %c -> Digit\n", ch);
        }
        else {
            printf("Input: %c -> Ignored\n", ch);
        }
    }

    printf("\n--- Final Result ---\n");

    if(strlen(password) >= 6 && hasLower && hasUpper && hasDigit) {
        printf("ACCEPTED\n");
    } else {
        printf("REJECTED\n");
    }

    return 0;
}