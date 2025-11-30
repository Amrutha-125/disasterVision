export type SentinelUrls = {
  s1Url: string; // Sentinel-1 (Radar) image URL
  s2Url: string; // Sentinel-2 (Optical) image URL
};

// Refactored to fetch both images simultaneously
export const getSentinelImageUrls = async (
  lat: number,
  lon: number
): Promise<SentinelUrls> => {
  /* 1️⃣ Fetch token from your backend */
  const tokenRes = await fetch("/api/sentinel-token", { method: "POST" });
  if (!tokenRes.ok) throw new Error("Failed to fetch Sentinel token");
  const { token } = await tokenRes.json();

  /* 2️⃣ Prepare dynamic date range (last 60 days) */
  const toDate = new Date();
  const fromDate = new Date();
  fromDate.setDate(toDate.getDate() - 60);
  const toISO = toDate.toISOString();
  const fromISO = fromDate.toISOString();

  const bbox = [lon - 0.05, lat - 0.05, lon + 0.05, lat + 0.05];
  const commonBodyParts = {
    input: {
      bounds: {
        bbox: bbox,
        properties: { crs: "http://www.opengis.net/def/crs/EPSG/0/4326" },
      },
    },
    output: {
      width: 512,
      height: 512,
      responses: [{ identifier: "default", format: { type: "image/png" } }],
    },
  };

  /* 3️⃣ Define Sentinel-1 (S1) Request Body */
  const s1RequestBody = {
    ...commonBodyParts,
    input: {
      ...commonBodyParts.input,
      data: [
        {
          type: "S1GRD", // Sentinel-1 GRD
          dataFilter: { timeRange: { from: fromISO, to: toISO } },
        },
      ],
    },
    evalscript: `//VERSION=3
function setup() {
  return {
    input: ["VV"],
    output: { bands: 3 }
  };
}
function evaluatePixel(sample) {
  let val = sample.VV * 2.5;
  return [val, val, val];
}`,
  };

  /* 4️⃣ Define Sentinel-2 (S2) Request Body */
  const s2RequestBody = {
    ...commonBodyParts,
    input: {
      ...commonBodyParts.input,
      data: [
        {
          type: "S2L2A", // Sentinel-2 L2A (Surface Reflectance)
          dataFilter: { 
            timeRange: { from: fromISO, to: toISO }, 
            maxCloudCoverage: 10 // Filter for low cloud cover
          },
        },
      ],
    },
    evalscript: `//VERSION=3
function setup() {
  return {
    input: ["B04", "B03", "B02"], // RGB True Color
    output: { bands: 3 }
  };
}
function evaluatePixel(sample) {
  return [sample.B04 * 2.5, sample.B03 * 2.5, sample.B02 * 2.5];
}`,
  };

  /* 5️⃣ Execute both requests concurrently */
  const url = "https://services.sentinel-hub.com/api/v1/process";
  const headers = {
    Authorization: `Bearer ${token}`,
    "Content-Type": "application/json",
    Accept: "image/png",
  };

  const s1Promise = fetch(url, {
    method: "POST",
    headers: headers,
    body: JSON.stringify(s1RequestBody),
  });

  const s2Promise = fetch(url, {
    method: "POST",
    headers: headers,
    body: JSON.stringify(s2RequestBody),
  });

  // Use Promise.allSettled to wait for both to complete
  const [s1Result, s2Result] = await Promise.allSettled([s1Promise, s2Promise]);

  const processResponse = async (
    result: PromiseSettledResult<Response>,
    satellite: string
  ): Promise<string> => {
    if (result.status === "rejected") {
      throw new Error(`Failed to fetch ${satellite} image: ${result.reason}`);
    }

    const response = result.value;
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Sentinel Hub ${satellite} request failed: ${errorText}`);
    }

    const blob = await response.blob();
    return URL.createObjectURL(blob);
  };

  // Process the results
  const s1Url = await processResponse(s1Result, "Sentinel-1");
  const s2Url = await processResponse(s2Result, "Sentinel-2");

  return { s1Url, s2Url };
};

