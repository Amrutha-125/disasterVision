"use client";

import { Country, State, City } from "country-state-city";
import type { IState, ICity } from "country-state-city";
import { useEffect, useState } from "react";

interface Props {
  countryCode: string;
  setCountryCode: (c: string) => void;
  stateCode: string;
  setStateCode: (s: string) => void;
  cityName: string;
  setCityName: (c: string) => void;
}

export default function LocationSelector({
  countryCode,
  setCountryCode,
  stateCode,
  setStateCode,
  cityName,
  setCityName,
}: Props) {
  const [states, setStates] = useState<IState[]>([]);
  const [cities, setCities] = useState<ICity[]>([]);

  /* fetch states when country changes */
  useEffect(() => {
    setStates(countryCode ? State.getStatesOfCountry(countryCode) : []);
    setStateCode("");
    setCityName("");
  }, [countryCode]);

  /* fetch cities when state changes */
  useEffect(() => {
    setCities(stateCode ? City.getCitiesOfState(countryCode, stateCode) : []);
    setCityName("");
  }, [stateCode]);

  const selectCls =
    "w-full p-3 rounded bg-gray-800 text-white border border-gray-600";

  return (
    <div className="grid md:grid-cols-3 gap-4">
      <select
        className={selectCls}
        value={countryCode}
        onChange={(e) => setCountryCode(e.target.value)}
        title="Country"
      >
        <option value="">Select Country</option>
        {Country.getAllCountries().map((c) => (
          <option key={c.isoCode} value={c.isoCode}>
            {c.name}
          </option>
        ))}
      </select>

      <select
        className={selectCls}
        value={stateCode}
        onChange={(e) => setStateCode(e.target.value)}
        disabled={!states.length}
        title="State"
      >
        <option value="">Select State</option>
        {states.map((s) => (
          <option key={s.isoCode} value={s.isoCode}>
            {s.name}
          </option>
        ))}
      </select>

      <select
        className={selectCls}
        value={cityName}
        onChange={(e) => setCityName(e.target.value)}
        disabled={!cities.length}
        title="City"
      >
        <option value="">Select City</option>
        {cities.map((city) => (
          <option key={city.name} value={city.name}>
            {city.name}
          </option>
        ))}
      </select>
    </div>
  );
}
