import rtmidi
import liblo
server = liblo.Server(10001)
midiout = rtmidi.MidiOut()
midiout.open_virtual_port("My virtual output")

def noteon_cb(path, args):
    i = args[0]
    note_on = [0x90, int(i)+32, 100]
    midiout.send_message( note_on )

def noteoff_cb(path, args):
    i = args[0]
    note_off = [0x80, int(i), 0 ] 
    midiout.send_message( note_off )

server.add_method("/noteon", 'i', noteon_cb)
server.add_method("/noteoff", 'i', noteoff_cb)
while True:
    server.recv(100)
