// Reconstructed 022 typed executor. Engineered semantics, not learned memory.
#include <cstdint>
#include <vector>
#include <omp.h>
struct Op {int64_t op,dst,src,value,guard;};
static int64_t read(const int64_t*k,const int64_t*d,int64_t n,int64_t i){
 for(int64_t t=0;t<n;++t){if(k[i]==0)return -1;if(k[i]==1)return d[i];i=d[i];}return -1;
}
static void step(int64_t*k,int64_t*d,int64_t n,const Op&o,int64_t&ans){
 ans=-2;switch(o.op){case 0:break;case 1:k[o.dst]=1;d[o.dst]=o.value;break;
 case 2:{auto v=read(k,d,n,o.src);k[o.dst]=v<0?0:1;d[o.dst]=v<0?0:v;break;}
 case 3:k[o.dst]=2;d[o.dst]=o.src;break;case 4:k[o.dst]=0;d[o.dst]=0;break;
 case 5:ans=read(k,d,n,o.dst);break;case 6:if(read(k,d,n,o.src)==o.guard){k[o.dst]=1;d[o.dst]=o.value;}break;}
}
extern "C" int execute(int64_t*k,int64_t*d,int64_t n,const int64_t*raw,int64_t t,const int64_t*bank_ids,int banks,int threads,int parallel,int64_t*ans,int64_t*stats){
 if(n<1||t<0||banks<1||banks>n||threads<1)return 1;
 auto bank=[=](int64_t i){return int(bank_ids[i]);}; bool cross=false;
 for(int64_t i=0;i<n;++i)if(bank_ids[i]<0||bank_ids[i]>=banks)return 1;
 for(int64_t i=0;i<n;++i){if(k[i]<0||k[i]>2)return 2;if(k[i]==0&&d[i]!=0)return 2;if(k[i]==1&&(d[i]<0||d[i]>4294967295))return 2;if(k[i]==2){if(d[i]<0||d[i]>=n)return 2;if(bank(i)!=bank(d[i]))cross=true;}}
 // Validate whole input before mutation, including inactive conditional operands.
 for(int64_t i=0;i<t;++i){const auto*p=raw+5*i;if(p[0]<0||p[0]>6||p[1]<0||p[1]>=n)return 3;if((p[0]==2||p[0]==3||p[0]==6)&&(p[2]<0||p[2]>=n))return 3;if((p[0]==1||p[0]==6)&&(p[3]<0||p[3]>4294967295))return 3;if(p[0]==6&&(p[4]<0||p[4]>4294967295))return 3;if((p[0]==2||p[0]==3||p[0]==6)&&bank(p[1])!=bank(p[2]))cross=true;}
 stats[0]=cross;stats[1]=0;stats[2]=banks;
 if(!parallel||cross||banks==1){for(int64_t i=0;i<t;++i){const auto*p=raw+5*i;step(k,d,n,Op{p[0],p[1],p[2],p[3],p[4]},ans[i]);}return 0;}
 std::vector<std::vector<int64_t>> jobs(banks);
 for(int64_t i=0;i<t;++i)jobs[bank(raw[5*i+1])].push_back(i);
 int team=1;
 #pragma omp parallel num_threads(threads)
 {
 #pragma omp single
 team=omp_get_num_threads();
 #pragma omp for schedule(static)
 for(int b=0;b<banks;++b){for(auto i:jobs[b]){const auto*p=raw+5*i;step(k,d,n,Op{p[0],p[1],p[2],p[3],p[4]},ans[i]);}}
 }
 stats[1]=team;return 0;
}
