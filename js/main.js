(function() {
	var video = document.querySelector('video');

	var pictureWidth = 640;
	var pictureHeight = 360;

	var fxCanvas = null;
	var texture = null;

	function checkRequirements() {
		var deferred = new $.Deferred();

		// Check if getUserMedia is available
		if (!Modernizr.getusermedia) {
			deferred
					.reject('Your browser doesn\'t support getUserMedia (according to Modernizr).');
		}

		// Check if WebGL is available
		if (Modernizr.webgl) {
			try {
				// setup glfx.js
				fxCanvas = fx.canvas();
			} catch (e) {
				deferred
						.reject('Sorry, glfx.js failed to initialize. WebGL issues?');
			}
		} else {
			deferred
					.reject('Your browser doesn\'t support WebGL (according to Modernizr).');
		}

		deferred.resolve();

		return deferred.promise();
	}

	function searchForRearCamera() {
		var deferred = new $.Deferred();

		// MediaStreamTrack.getSources seams to be supported only by Chrome
		if (MediaStreamTrack && MediaStreamTrack.getSources) {
			MediaStreamTrack
					.getSources(function(sources) {
						var rearCameraIds = sources
								.filter(
										function(source) {
											return (source.kind === 'video' && source.facing === 'environment');
										}).map(function(source) {
									return source.id;
								});

						if (rearCameraIds.length) {
							deferred.resolve(rearCameraIds[0]);
						} else {
							deferred.resolve(null);
						}
					});
		} else {
			deferred.resolve(null);
		}

		return deferred.promise();
	}

	function setupVideo(rearCameraId) {
		var deferred = new $.Deferred();
		var getUserMedia = Modernizr.prefixed('getUserMedia', navigator);
		var videoSettings = {
			video : {
				optional : [ {
					width : {
						min : pictureWidth
					}
				}, {
					height : {
						min : pictureHeight
					}
				} ]
			}
		};

		// if rear camera is available - use it
		if (rearCameraId) {
			videoSettings.video.optional.push({
				sourceId : rearCameraId
			});
		}

		getUserMedia(videoSettings, function(stream) {
			// Setup the video stream
			video.src = window.URL.createObjectURL(stream);

			window.stream = stream;

			video.addEventListener("loadedmetadata", function(e) {
				// get video width and height as it might be different
				// than we requested
				pictureWidth = this.videoWidth;
				pictureHeight = this.videoHeight;

				if (!pictureWidth && !pictureHeight) {
					// firefox fails to deliver info about video size on
					// time (issue #926753), we have to wait
					var waitingForSize = setInterval(function() {
						if (video.videoWidth && video.videoHeight) {
							pictureWidth = video.videoWidth;
							pictureHeight = video.videoHeight;

							clearInterval(waitingForSize);
							deferred.resolve();
						}
					}, 100);
				} else {
					deferred.resolve();
				}
			}, false);
		}, function() {
			deferred.reject('打不开你的摄像头我也很无奈，可以右上角在浏览器中打开');
		});

		return deferred.promise();
	}

	function step1() {
		checkRequirements().then(searchForRearCamera).then(setupVideo).done(
				function() {
					// Enable the 'take picture' button
					$('#takePicture').removeAttr('disabled');
					// Hide the 'enable the camera' info
					$('#step1 figure').removeClass('not-ready');
				}).fail(function(error) {
			showError(error);
		});
	}

	function uploadPic(data) {
		data = data.split(',')[1];
		data = window.atob(data);
		var ia = new Uint8Array(data.length);
		for (var i = 0; i < data.length; i++) {
			ia[i] = data.charCodeAt(i);
		}
		var blob = new Blob([ ia ], {
			type : "image/png"
		});
		var fd = new FormData();
		fd.append('file', blob);
		$.ajax({
			url : "higary.iask.in/realTimeIR/uploadPic",
			async : false,
			cache : false,
			contentType : false,
			processData : false,
			type : "POST",
			data : fd,
			dataType : "json",
			success : function(data) {
				$("#result").empty();
				$("#result").text("识别结果：" + data.result);
			}
		});
	}

	function step2() {
		var canvas = document.querySelector('#step2 canvas');
		var img = document.querySelector('#step2 img');

		// setup canvas
		canvas.width = pictureWidth;
		canvas.height = pictureHeight;

		var ctx = canvas.getContext('2d');

		// draw picture from video on canvas
		ctx.drawImage(video, 0, 0);

		// modify the picture using glfx.js filters
		texture = fxCanvas.texture(canvas);
		fxCanvas.draw(texture).update();

		window.texture = texture;
		window.fxCanvas = fxCanvas;

		$(img)
		// show output from glfx.js
		.attr('src', fxCanvas.toDataURL());

		uploadPic(fxCanvas.toDataURL());
	}

	/***************************************************************************
	 * UI Stuff
	 **************************************************************************/

	// start step1 immediately
	step1();

	function changeStep(step) {
		if (step === 1) {
			video.play();
		} else {
			video.pause();
		}

		$('body').attr('class', 'step' + step);
	}

	function showError(text) {
		$('.alert').show().find('span').text(text);
	}

	$('#takePicture').click(function() {
		step2();
		changeStep(2);
	});

	$('#go-back').click(function() {
		step1();
		changeStep(1);
	});
})();