#include <iostream>
#include <vector>
#include <string>

using namespace std;


/*----------------------------------------------------------
Using C++ on Linux in VS Code
From: https://code.visualstudio.com/docs/cpp/config-linux
Date: 2022-3-21
---------------------------------------------------------*/


int main()
{
    vector<string> msg {"Hello", "C++", "World", "from", "VS Code", "and the C++ extension!"};

    for (const string& word : msg)
    {
        cout << word << " ";
    }
    cout << endl;
}
