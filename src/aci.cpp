#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <limits>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <vector>
#include <chrono>

constexpr int D=128;
struct Posting {uint64_t key; uint32_t id;};
struct Index {
 int n, groups, group_size, bins;
 std::vector<float> data, lo, step;
 std::vector<uint8_t> codes;
 std::vector<std::unordered_map<uint64_t,std::vector<uint32_t>>> lists;
 Index(const float* x,int n_,int gs,int b):n(n_),groups(D/gs),group_size(gs),bins(b),data(x,x+(size_t)n_*D),lo(D),step(D),codes((size_t)n_*D),lists(groups){
  for(int j=0;j<D;j++){
   float mn=std::numeric_limits<float>::infinity(),mx=-mn;
   for(int i=0;i<n;i++){float v=data[(size_t)i*D+j];mn=std::min(mn,v);mx=std::max(mx,v);}
   lo[j]=mn;step[j]=std::max((mx-mn)/bins,1e-12f);
  }
  for(int i=0;i<n;i++){
   for(int j=0;j<D;j++){
    int z=std::clamp((int)((data[(size_t)i*D+j]-lo[j])/step[j]),0,bins-1);
    codes[(size_t)i*D+j]=(uint8_t)z;
   }
   for(int g=0;g<groups;g++){
    uint64_t key=0;
    for(int t=0;t<group_size;t++)key=key*bins+codes[(size_t)i*D+g*group_size+t];
    lists[g][key].push_back(i);
   }
  }
 }
};
static inline float dist128(const float* a,const float* b,float cutoff){
 float total=0;
 for(int j=0;j<D;j+=16){
  __m256 a0=_mm256_loadu_ps(a+j),b0=_mm256_loadu_ps(b+j);
  __m256 a1=_mm256_loadu_ps(a+j+8),b1=_mm256_loadu_ps(b+j+8);
  __m256 v0=_mm256_sub_ps(a0,b0),v1=_mm256_sub_ps(a1,b1);
  __m256 sq=_mm256_add_ps(_mm256_mul_ps(v0,v0),_mm256_mul_ps(v1,v1));
  alignas(32) float tmp[8];_mm256_store_ps(tmp,sq);
  for(float v:tmp)total+=v;
  if(total>cutoff)return total;
 }
 return total;
}
struct Cell {float bound;int g;std::array<uint8_t,8> b;};
struct Compare {bool operator()(const Cell&a,const Cell&b)const{return a.bound>b.bound;}};
static uint64_t encode(const Cell& c,int gs,int bins){uint64_t key=0;for(int t=0;t<gs;t++)key=key*bins+c.b[t];return key;}
static float bound(const Index&idx,const float*q,const Cell&c){
 float s=0;for(int t=0;t<idx.group_size;t++){
  int j=c.g*idx.group_size+t;float l=idx.lo[j]+c.b[t]*idx.step[j];float u=l+idx.step[j];
  float z=q[j]<l?l-q[j]:(q[j]>u?q[j]-u:0.f);s+=z*z;
 }return s;
}
extern "C" {
void* aci_create(const float*x,int n,int group_size,int bins){
 if(!x||n<=0||group_size<1||group_size>8||D%group_size||bins<2||bins>256)return nullptr;
 try{return new Index(x,n,group_size,bins);}catch(...){return nullptr;}
}
void aci_destroy(void*p){delete static_cast<Index*>(p);}
size_t aci_index_bytes(void*p){
 if(!p)return 0;auto&ix=*static_cast<Index*>(p);size_t s=ix.data.capacity()*sizeof(float)+ix.codes.capacity()+ix.lo.capacity()*sizeof(float)+ix.step.capacity()*sizeof(float);
 for(auto&g:ix.lists){s+=g.bucket_count()*sizeof(void*);for(auto&kv:g)s+=sizeof(kv)+kv.second.capacity()*sizeof(uint32_t)+sizeof(void*)*2;}
 return s;
}
int aci_search(void*p,const float*q,int k,int posting_budget,int rerank_budget,int*out,float*distance,int*stats){
 if(!p||!q||!out||!distance||!stats||k<1||posting_budget<1||rerank_budget<1)return -1;
 auto&ix=*static_cast<Index*>(p);
 std::priority_queue<Cell,std::vector<Cell>,Compare> frontier;
 for(int g=0;g<ix.groups;g++){
  Cell c{};c.g=g;
  for(int t=0;t<ix.group_size;t++){
   int j=g*ix.group_size+t;c.b[t]=std::clamp((int)((q[j]-ix.lo[j])/ix.step[j]),0,ix.bins-1);
  }
  c.bound=bound(ix,q,c);frontier.push(c);
 }
 std::vector<std::unordered_map<uint64_t,bool>> visited(ix.groups);
 std::unordered_map<uint32_t,std::pair<int,float>> candidates;
 int read=0,probes=0;
 int max_probes=std::min(100000,std::max(1024,posting_budget*2));
 while(!frontier.empty()&&read<posting_budget&&probes<max_probes){
  Cell c=frontier.top();frontier.pop();uint64_t key=encode(c,ix.group_size,ix.bins);
  if(!visited[c.g].emplace(key,true).second)continue;
  ++probes;
  auto it=ix.lists[c.g].find(key);
  if(it!=ix.lists[c.g].end()){
   const auto& ids=it->second;
   int take=std::min((int)ids.size(),posting_budget-read);
   for(int z=0;z<take;z++){
    auto&v=candidates[ids[z]];v.first++;v.second+=c.bound;
   }
   read+=take;
  }
  for(int t=0;t<ix.group_size;t++)for(int direction:{-1,1}){
   int nb=(int)c.b[t]+direction;if(nb<0||nb>=ix.bins)continue;
   Cell next=c;next.b[t]=(uint8_t)nb;
   uint64_t nk=encode(next,ix.group_size,ix.bins);
   if(visited[c.g].find(nk)==visited[c.g].end()){
    next.bound=bound(ix,q,next);frontier.push(next);
   }
  }
 }
 std::vector<std::pair<uint32_t,std::pair<int,float>>> ranked(candidates.begin(),candidates.end());
 auto better=[](const auto&a,const auto&b){
  if(a.second.first!=b.second.first)return a.second.first>b.second.first;
  if(a.second.second!=b.second.second)return a.second.second<b.second.second;
  return a.first<b.first;
 };
 int take=std::min((int)ranked.size(),std::max(k,rerank_budget));
 if(take<(int)ranked.size())std::partial_sort(ranked.begin(),ranked.begin()+take,ranked.end(),better);
 else std::sort(ranked.begin(),ranked.end(),better);
 using Hit=std::pair<float,uint32_t>;
 std::priority_queue<Hit> best;
 for(int i=0;i<take;i++){
  uint32_t id=ranked[i].first;
  float cutoff=(int)best.size()<k?std::numeric_limits<float>::infinity():best.top().first;
  float d=dist128(q,ix.data.data()+(size_t)id*D,cutoff);
  if((int)best.size()<k)best.emplace(d,id);
  else if(d<best.top().first||(d==best.top().first&&id<best.top().second)){best.pop();best.emplace(d,id);}
 }
 int count=best.size();
 for(int i=count-1;i>=0;i--){out[i]=best.top().second;distance[i]=best.top().first;best.pop();}
 for(int i=count;i<k;i++){out[i]=-1;distance[i]=std::numeric_limits<float>::infinity();}
 stats[0]=read;stats[1]=candidates.size();stats[2]=take;stats[3]=probes;
 return count;
}
}
