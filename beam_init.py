# -*-coding:utf-8-*-
import math as m
import matplotlib.pyplot as plt
import numpy as np


def setInitBeamCenterPos(i, SateBLH, type):
    if type == "IRIDIUM":
        # 波束的数量以及功率大小
        numPointsofC1 = 3
        numPointsofC2 = 9
        numPointsofC3 = 15
        numPointsofC4 = 21
        # 设定每一圈里中心距离orbit inclination : 86.402 degree, 1.508 rad
        maxArcDistFromSubPos = 2000000  # 2000km：单星覆盖arc长度
        arcDist1 = maxArcDistFromSubPos * 1 / 4
        arcDist2 = maxArcDistFromSubPos * 2 / 4
        arcDist3 = maxArcDistFromSubPos * 7 / 8
        # arcDist4 = maxArcDistFromSubPos * 7 / 8
        # 设置每一层
        allocate1, lat_log1 = createInitBeamCenterPos(i, 0, SateBLH, numPointsofC1, arcDist1)
        allocate2, lat_log2 = createInitBeamCenterPos(i, numPointsofC1, SateBLH, numPointsofC2, arcDist2)
        allocate3, lat_log3 = createInitBeamCenterPos(i, numPointsofC1 + numPointsofC2, SateBLH, numPointsofC3,
                                                      arcDist3)
        # allocate4, lat_log4 = createInitBeamCenterPos(i, numPointsofC1 + numPointsofC2 + numPointsofC3, SateBLH,
        #                                               numPointsofC4, arcDist4)
        beam = np.array(allocate1 + allocate2+allocate3)
        lat_log = np.array(lat_log1 + lat_log2+lat_log3)
    return beam, lat_log


def createInitBeamCenterPos(satnum, numPointsof, SateBLH, numPoints, arcDist):
    beamCenterAlloc = []
    beamCenterAlloc_lat_log = []
    for i in range(numPoints):
        # 方位角设计
        cellid = satnum * 48 + numPointsof + i
        azimuth = (i + 1) * 360.0 / numPoints
        # print("azimuth", azimuth)
        a = 6378137.0
        b = 6356752.3142
        f = 1.0 / 298.257223563
        alpha1 = azimuth * m.pi / 180.0
        sinAlpha1 = m.sin(alpha1)
        cosAlpha1 = m.cos(alpha1)
        tanU1 = (1 - f) * m.tan(SateBLH[0] * m.pi / 180.0)
        cosU1 = 1 / m.sqrt((1 + tanU1 * tanU1))
        sinU1 = tanU1 * cosU1
        sigma1 = m.atan2(tanU1, cosAlpha1)
        sinAlpha = cosU1 * sinAlpha1
        cosSqAlpha = 1 - sinAlpha * sinAlpha
        uSq = cosSqAlpha * (a * a - b * b) / (b * b)
        A = 1 + uSq / 16384 * (4096 + uSq * (-768 + uSq * (320 - 175 * uSq)))
        B = uSq / 1024 * (256 + uSq * (-128 + uSq * (74 - 47 * uSq)))

        sigma = arcDist / (b * A)
        sigmaP = 2 * m.pi
        sinSigma = m.sin(sigma)
        cosSigma = m.cos(sigma)
        cos2SigmaM = m.cos(2 * sigma1 + sigma)
        while (abs(sigma - sigmaP) > 1e-12):
            deltaSigma = B * sinSigma * (cos2SigmaM + B / 4 * (cosSigma * (-1 + 2 * cos2SigmaM * cos2SigmaM) -
                                                               B / 6 * cos2SigmaM * (-3 + 4 * sinSigma * sinSigma) * (
                                                                       -3 + 4 * cos2SigmaM * cos2SigmaM)))
            sigmaP = sigma
            sigma = arcDist / (b * A) + deltaSigma
        tmp = sinU1 * sinSigma - cosU1 * cosSigma * cosAlpha1
        lat2 = m.atan(
            (sinU1 * cosSigma + cosU1 * sinSigma * cosAlpha1) / ((1 - f) * m.sqrt(sinAlpha * sinAlpha + tmp * tmp)))
        lambd = m.atan((sinSigma * sinAlpha1) / (cosU1 * cosSigma - sinU1 * sinSigma * cosAlpha1))
        C = f / 16 * cosSqAlpha * (4 + f * (4 - 3 * cosSqAlpha))
        L = lambd - (1 - C) * f * sinAlpha * (
                sigma + C * sinSigma * (cos2SigmaM + C * cosSigma * (-1 + 2 * cos2SigmaM * cos2SigmaM)))
        pointPosition = GeographicToCartesianCoordinates(lat2 * 180 / m.pi, SateBLH[1] + L * 180 / m.pi, 0, "GRS80")
        lat_log = ConstructFromVector(pointPosition[0], pointPosition[1], pointPosition[2], "GRS80")
        beamCenterAlloc.append([cellid, pointPosition[0], pointPosition[1], pointPosition[2]])
        beamCenterAlloc_lat_log.append([cellid, lat_log[0], lat_log[1], lat_log[2]])
    return beamCenterAlloc, beamCenterAlloc_lat_log


def ConstructFromVector(x, y, z, sphType):
    # a: semi - major axis of earth
    # e: first eccentricity of earth
    EARTH_RADIUS = 6371e3
    EARTH_SEMIMAJOR_AXIS = 6378137
    EARTH_GRS80_ECCENTRICITY = 0.0818191910428158
    EARTH_WGS84_ECCENTRICITY = 0.0818191908426215
    if sphType == "SPHERE":
        a = EARTH_RADIUS
        e = 0
    if sphType == "GRS80":
        a = EARTH_SEMIMAJOR_AXIS
        e = EARTH_GRS80_ECCENTRICITY
    else:  # if sphType == WGS84
        a = EARTH_SEMIMAJOR_AXIS
        e = EARTH_WGS84_ECCENTRICITY

    latitudeRadians = m.asin(z / m.sqrt(x ** 2 + y ** 2 + z ** 2))
    latitude = latitudeRadians * 180 / m.pi

    if x == 0 and y > 0:
        longitude = 90
    elif x == 0 and y < 0:
        longitude = -90
    elif x < 0 and y >= 0:
        longitudeRadians = m.atan(y / x)
        longitude = longitudeRadians * 180 / m.pi + 180
    elif x < 0 and y <= 0:
        longitudeRadians = m.atan(y / x)
        longitude = longitudeRadians * 180 / m.pi - 180
    else:
        longitudeRadians = m.atan(y / x)
        longitude = longitudeRadians * 180 / m.pi

    Rn = a / (m.sqrt(1 - pow(e, 2) * pow(m.sin(latitudeRadians), 2)))
    altitude = m.sqrt(x ** 2 + y ** 2 + z ** 2) - Rn
    # print("altitude", altitude)
    return [latitude, longitude, altitude]


def GeographicToCartesianCoordinates(latitude, longitude, altitude, sphType):
    latitudeRadians = latitude * m.pi / 180
    longitudeRadians = longitude * m.pi / 180
    # print("longitudeRadians", longitudeRadians)
    # print("latitudeRadians", latitudeRadians)
    # a: semi - major axis of earth
    # e: first eccentricity of earth
    EARTH_RADIUS = 6371e3
    EARTH_GRS80_ECCENTRICITY = 0.0818191910428158
    EARTH_WGS84_ECCENTRICITY = 0.0818191908426215
    EARTH_SEMIMAJOR_AXIS = 6378137
    EARTH_SEMIMAJOR_BXIS = 6356752.3142451793
    if sphType == "SPHERE":
        a = EARTH_RADIUS
        e = 0
    if sphType == "GRS80":
        a = EARTH_SEMIMAJOR_AXIS
        e = EARTH_GRS80_ECCENTRICITY
    else:  # if sphType == WGS84
        a = EARTH_SEMIMAJOR_AXIS
        e = EARTH_WGS84_ECCENTRICITY
    Rn = a / (m.sqrt(1 - pow(e, 2) * pow(m.sin(latitudeRadians), 2)))  # radius of  curvature
    # print("rn", Rn)
    x = (Rn + altitude) * m.cos(latitudeRadians) * m.cos(longitudeRadians)
    y = (Rn + altitude) * m.cos(latitudeRadians) * m.sin(longitudeRadians)
    z = (Rn + altitude) * m.sin(latitudeRadians)
    # z = ((1 - pow(e, 2)) * Rn + altitude) * m.sin(latitudeRadians)
    cartesianCoordinates = [x, y, z]
    return cartesianCoordinates


def position_plot(beam_lat_log, user_lat):
    assert type(beam_lat_log) is np.ndarray and type(user_lat) is np.ndarray, '输入类型不为numpy'
    fig, ax = plt.subplots(1, 1)
    beam_x = beam_lat_log[:, 1]
    beam_y = beam_lat_log[:, 2]
    user_x = user_lat[:, 0]
    user_y = user_lat[:, 1]
    ax.scatter(beam_x, beam_y, s=20, c='b')
    ax.scatter(user_x, user_y, s=5, c='r')
    plt.show()
    plt.close()


def userconnectsate(userposition, beamposition, request, usernumer):
    all_connect_info = []
    beam_number = np.zeros(usernumer)
    # print(len(beam_number))
    # input()
    for i in range(len(userposition)):
        user = userposition[i]
        distance_max = np.inf
        connect_beam_position = 0
        for j in range(len(beamposition)):
            beam = beamposition[j][1:]
            distance = np.sqrt(np.sum((user - beam) ** 2))
            if distance < distance_max:
                distance_max = distance
                connect_beam_position = beam
                beam_number[request[i]] = beamposition[j][0] + 1
        user_connect_info = np.hstack((request[i], user, connect_beam_position, [0, 0, 0],beam_number[request[i]]))
        all_connect_info.append(user_connect_info)
    all_connect_info = np.array(all_connect_info)

    return all_connect_info, beam_number

    return 0


if __name__ == '__main__':
    beam, lat_log = setInitBeamCenterPos(0, [0, 0, 0], type='IRIDIUM')
    print(40008080.0 / 20.0)
