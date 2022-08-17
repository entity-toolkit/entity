#define GET_MACRO(_1, _2, _3, NAME, ...) NAME

#define BX1(...)                         GET_MACRO(__VA_ARGS__, BX1_3D, BX1_2D, BX1_1D, )(__VA_ARGS__)
#define BX1_1D(I)                        (m_mblock.em((I), em::bx1))
#define BX1_2D(I, J)                     (m_mblock.em((I), (J), em::bx1))
#define BX1_3D(I, J, K)                  (m_mblock.em((I), (J), (K), em::bx1))

#define BX2(...)                         GET_MACRO(__VA_ARGS__, BX2_3D, BX2_2D, BX2_1D, )(__VA_ARGS__)
#define BX2_1D(I)                        (m_mblock.em((I), em::bx2))
#define BX2_2D(I, J)                     (m_mblock.em((I), (J), em::bx2))
#define BX2_3D(I, J, K)                  (m_mblock.em((I), (J), (K), em::bx2))

#define BX3(...)                         GET_MACRO(__VA_ARGS__, BX3_3D, BX3_2D, BX3_1D, )(__VA_ARGS__)
#define BX3_1D(I)                        (m_mblock.em((I), em::bx3))
#define BX3_2D(I, J)                     (m_mblock.em((I), (J), em::bx3))
#define BX3_3D(I, J, K)                  (m_mblock.em((I), (J), (K), em::bx3))

#define EX1(...)                         GET_MACRO(__VA_ARGS__, EX1_3D, EX1_2D, EX1_1D, )(__VA_ARGS__)
#define EX1_1D(I)                        (m_mblock.em((I), em::ex1))
#define EX1_2D(I, J)                     (m_mblock.em((I), (J), em::ex1))
#define EX1_3D(I, J, K)                  (m_mblock.em((I), (J), (K), em::ex1))

#define EX2(...)                         GET_MACRO(__VA_ARGS__, EX2_3D, EX2_2D, EX2_1D, )(__VA_ARGS__)
#define EX2_1D(I)                        (m_mblock.em((I), em::ex2))
#define EX2_2D(I, J)                     (m_mblock.em((I), (J), em::ex2))
#define EX2_3D(I, J, K)                  (m_mblock.em((I), (J), (K), em::ex2))

#define EX3(...)                         GET_MACRO(__VA_ARGS__, EX3_3D, EX3_2D, EX3_1D, )(__VA_ARGS__)
#define EX3_1D(I)                        (m_mblock.em((I), em::ex3))
#define EX3_2D(I, J)                     (m_mblock.em((I), (J), em::ex3))
#define EX3_3D(I, J, K)                  (m_mblock.em((I), (J), (K), em::ex3))

#define JX1(...)                         GET_MACRO(__VA_ARGS__, JX1_3D, JX1_2D, JX1_1D, )(__VA_ARGS__)
#define JX1_1D(I)                        (m_mblock.cur((I), cur::jx1))
#define JX1_2D(I, J)                     (m_mblock.cur((I), (J), cur::jx1))
#define JX1_3D(I, J, K)                  (m_mblock.cur((I), (J), (K), cur::jx1))

#define JX2(...)                         GET_MACRO(__VA_ARGS__, JX2_3D, JX2_2D, JX2_1D, )(__VA_ARGS__)
#define JX2_1D(I)                        (m_mblock.cur((I), cur::jx2))
#define JX2_2D(I, J)                     (m_mblock.cur((I), (J), cur::jx2))
#define JX2_3D(I, J, K)                  (m_mblock.cur((I), (J), (K), cur::jx2))

#define JX3(...)                         GET_MACRO(__VA_ARGS__, JX3_3D, JX3_2D, JX3_1D, )(__VA_ARGS__)
#define JX3_1D(I)                        (m_mblock.cur((I), cur::jx3))
#define JX3_2D(I, J)                     (m_mblock.cur((I), (J), cur::jx3))
#define JX3_3D(I, J, K)                  (m_mblock.cur((I), (J), (K), cur::jx3))


// regex

// find: m_mblock.em\((.*?), em::bx(.*?)\)
// replace: BX$2($1)

// find: m_mblock.em\((.*?), em::ex(.*?)\)
// replace: EX$2($1)

// find: m_mblock.cur\((.*?), cur::jx(.*?)\)
// replace: JX$2($1)