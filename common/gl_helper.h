#ifndef __GL_HELPER_H__
#define __GL_HELPER_H__

#include<GL/glut.h>
#include<GL/gtext.h>
#include<GL/glx.h>

#define GET_PROC_ADDRESS( str ) glXGetProcAddress( (const GLubyte *)str )

#endif