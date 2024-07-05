#include<stdio.h>
int GetNext(int j,char T[])
{
	if(j==0)return -1;
	if(j>0)
	{
		int k=GetNext(j-1,T);
		while(k>=0)
		{
			if(T[j-1]==T[k])return k+1;
			else k=GetNext(k,T);
		}
		return 0;
	}
	return 0;
}

int main(){
	char a[] = "ABCDABD";
	printf("%d", GetNext(6, a));
}