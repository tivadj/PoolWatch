module dllmain;

import std.c.windows.windows;
import core.sys.windows.dll;

__gshared HINSTANCE g_hInst;

version (PWDYNAMIC) // link as dinamic library (dll)
{
extern (Windows)
BOOL DllMain(HINSTANCE hInstance, ULONG ulReason, LPVOID pvReserved)
{
    final switch (ulReason)
    {
    case DLL_PROCESS_ATTACH:
        g_hInst = hInstance;
        dll_process_attach( hInstance, true );
        break;

    case DLL_PROCESS_DETACH:
        dll_process_detach( hInstance, true );
        break;

    case DLL_THREAD_ATTACH:
        dll_thread_attach( true, true );
        break;

    case DLL_THREAD_DETACH:
        dll_thread_detach( true, true );
        break;
    }
    return true;
}
}

// NOTE: main definition for static library is reuqired, otherwise deh_beg/deh_end are unresolved when linking.
// Or add '-main' flag to D compiler?
//void main(string[]) { }