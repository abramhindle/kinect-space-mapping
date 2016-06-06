s.boot;

~noteon = {
	arg msg;
	msg.postln;
	(freq: (msg[1].asInt() + 12).midicps).play();
};

OSCFunc.newMatching(~noteon, '/noteon');
